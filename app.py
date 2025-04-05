import os
import cv2
import qrcode
import pymongo
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
# Import Flask-SocketIO components
from flask_socketio import SocketIO, emit
from hashlib import sha256
import logging
from pyzbar.pyzbar import decode
import time
import base64
from io import BytesIO
from PIL import Image

# Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'qr_codes'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'face_data'), exist_ok=True)

# SocketIO setup
socketio = SocketIO(app)

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# MongoDB setup
try:
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    logger.info("MongoDB connection successful.")
    client.server_info()
    db = client["smart_attendance"]
    users_collection = db["users"]
    attendance_collection = db["attendance"]
    classes_collection = db["classes"]
    schedules_collection = db["schedules"]
    enrollments_collection = db["enrollments"]
except pymongo.errors.ConnectionError as e:
    logger.error(f"MongoDB connection failed: {e}")
    exit(1)

# Utility functions
def hash_password(password):
    return sha256(password.encode()).hexdigest()

def capture_face(user_id):
    # Try multiple camera indices to find an available one
    cap = None
    for index in range(3):  # Try indices 0, 1, 2
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap.isOpened():
            logger.info(f"Camera opened successfully at index {index}")
            break
        cap.release()
    else:
        logger.error("No available camera found!")
        flash("Unable to access camera! Ensure no other application is using it.")
        return None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_count = 0
    max_faces = 50
    face_paths = []
    last_capture_time = 0
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'face_data', f"user_{user_id}")
    os.makedirs(user_folder, exist_ok=True)

    try:
        while face_count < max_faces:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to grab frame from camera index {index}")
                time.sleep(0.1)  # Brief pause to avoid spamming
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            current_time = time.time()
            for (x, y, w, h) in faces:
                if current_time - last_capture_time >= 0.5:
                    face_path = os.path.join(user_folder, f"{face_count}.jpg")
                    cv2.imwrite(face_path, frame)
                    face_paths.append(face_path)
                    face_count += 1
                    last_capture_time = current_time
                    logger.debug(f"Captured face {face_count} for user {user_id}")
            cv2.imshow("Register Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logger.error(f"Error during face capture: {e}")
    finally:
        if cap and cap.isOpened():
            cap.release()
            logger.info("Camera released after face capture")
        cv2.destroyAllWindows()
    if face_count == 0:
        flash("No faces captured!")
        return None
    return face_paths

def scan_qr(frame):
    try:
        logger.debug("Attempting to scan QR code from frame")
        decoded = decode(frame)
        if decoded:
            logger.info(f"QR code detected: {decoded[0].data.decode('utf-8')}")
            return decoded[0].data.decode("utf-8")
        logger.debug("No QR code found in frame")
        return None
    except Exception as e:
        logger.error(f"Error decoding QR: {e}")
        return None

# Global variables for face verification
stored_features_cache = {}  # Cache for face features
verification_attempts = {}  # Track face verification attempts
detection_attempts = {}     # Track face detection attempts

def verify_face(stored_paths, live_frame):
    try:
        global stored_features_cache, verification_attempts, detection_attempts
        similarity_score = 0.0

        # Use cached features if available, otherwise compute and cache them
        if not stored_features_cache:
            logger.debug(f"Computing features for {len(stored_paths)} stored face images")
            stored_features_list = []
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Process only a subset of stored images for faster verification (max 10)
            sample_paths = stored_paths[:10] if len(stored_paths) > 10 else stored_paths
            valid_paths = 0

            for stored_path in sample_paths:
                if stored_path in stored_features_cache:
                    stored_features_list.append(stored_features_cache[stored_path])
                    valid_paths += 1
                    continue

                stored_img = cv2.imread(stored_path)
                if stored_img is None:
                    continue

                stored_gray = cv2.cvtColor(stored_img, cv2.COLOR_BGR2GRAY)
                stored_gray = cv2.equalizeHist(stored_gray)
                stored_faces = face_cascade.detectMultiScale(stored_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(stored_faces) == 0:
                    continue

                x, y, w, h = stored_faces[0]
                stored_face = stored_gray[y:y+h, x:x+w]
                features = extract_hog_features(stored_face)

                if features is not None:
                    stored_features_cache[stored_path] = features
                    stored_features_list.append(features)
                    valid_paths += 1

            logger.debug(f"Cached {valid_paths} face feature sets for faster verification")
        else:
            # Use cached features
            stored_features_list = list(stored_features_cache.values())
            logger.debug(f"Using {len(stored_features_list)} cached face feature sets")

        if not stored_features_list:
            logger.warning("No valid face data available")
            return False, 0.0

        # Process live frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Use balanced parameters for face detection - not too strict, not too lenient
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

        if len(faces) == 0:
            logger.debug("No face detected in frame or face too small")
            return False, 0.0

        # Try to match with the largest detected face (likely the closest to the camera)
        largest_face = max(faces, key=lambda face: face[2] * face[3])  # w * h
        x, y, w, h = largest_face

        # Check if face is large enough (close enough to camera)
        if w < 50 or h < 50:  # Increased from 40x40 to 50x50
            logger.debug(f"Face too small for reliable verification: {w}x{h} pixels")
            return False, 0.0

        # Calculate face visibility using image variance
        live_face = gray[y:y+h, x:x+w]
        face_variance = np.var(live_face)

        # If variance is too low, face might be covered or not visible clearly
        if face_variance < 300:  # Increased from 200 to 300
            logger.debug(f"Face visibility too low: variance = {face_variance}")
            return False, 0.0

        # Check face proportions (width/height ratio should be around 0.7-0.85 for a normal face)
        face_ratio = w / h
        if face_ratio < 0.5 or face_ratio > 1.2:  # Adjusted from 0.4-1.5 to 0.5-1.2
            logger.debug(f"Unusual face proportions detected: width/height = {face_ratio}")
            return False, 0.0

        # Extract features
        live_features = extract_hog_features(live_face)

        if live_features is not None:
            similarities = [compare_features(stored_features, live_features) for stored_features in stored_features_list]
            avg_similarity = np.mean(similarities)
            similarity_score = float(avg_similarity)  # Convert to float for JSON serialization

            # Set the threshold for recognition as requested
            threshold = 0.78  # Set to 0.78 as requested

            if avg_similarity > threshold:
                logger.info(f"Face verified successfully with similarity {avg_similarity}")
                return True, similarity_score
            else:
                logger.debug(f"Face verification failed - similarity {avg_similarity} below threshold {threshold}")
                return False, similarity_score
        else:
            logger.warning("Failed to extract HOG features from live frame")
            return False, 0.0
    except Exception as e:
        logger.error(f"Error in face verification: {e}")
        return False, 0.0

def extract_hog_features(face):
    try:
        # Apply histogram equalization to improve contrast
        face_eq = cv2.equalizeHist(face)

        # Apply Gaussian blur to reduce noise
        face_blur = cv2.GaussianBlur(face_eq, (5, 5), 0)

        # Resize to standard size for HOG
        face_resized = cv2.resize(face_blur, (64, 64))

        # Use HOG descriptor with standard parameters
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        features = hog.compute(face_resized)

        # Normalize features
        if features is not None and np.sum(features) != 0:
            features = features / (np.linalg.norm(features) + 1e-6)

        return features
    except Exception as e:
        logger.error(f"Error computing HOG features: {e}")
        return None

def compare_features(features1, features2):
    if features1 is None or features2 is None or features1.shape != features2.shape:
        return 0.0

    # Ensure features are normalized
    if np.sum(features1) != 0:
        features1 = features1 / (np.linalg.norm(features1) + 1e-6)
    if np.sum(features2) != 0:
        features2 = features2 / (np.linalg.norm(features2) + 1e-6)

    # Calculate cosine similarity
    similarity = np.dot(features1.flatten(), features2.flatten())

    # Ensure similarity is in [0, 1] range
    similarity = max(0.0, min(1.0, similarity))

    return similarity

def verify_gps(class_code, student_location):
    """Verify if the student is within 20 meters of the classroom location"""
    if not student_location:
        logger.warning("No student location provided for GPS verification")
        return False, None

    # Get the classroom location from the class data
    class_data = classes_collection.find_one({"code": class_code})
    if not class_data or "location" not in class_data:
        logger.warning(f"No location data found for class {class_code}")
        return True, None  # If no classroom location is set, bypass verification

    classroom_location = class_data["location"]
    classroom_lat = classroom_location["latitude"]
    classroom_lon = classroom_location["longitude"]

    student_lat = student_location["latitude"]
    student_lon = student_location["longitude"]

    # Calculate distance using Haversine formula
    R = 6371e3  # Earth radius in meters
    φ1, φ2 = np.radians(classroom_lat), np.radians(student_lat)
    Δφ = np.radians(student_lat - classroom_lat)
    Δλ = np.radians(student_lon - classroom_lon)
    a = np.sin(Δφ/2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ/2) ** 2
    c = 2 * np.atan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c

    # Maximum allowed distance: increased to 50 meters to account for GPS inaccuracy
    max_distance = 50  # Increased from 20m to 50m

    # Get accuracy information if available
    student_accuracy = student_location.get("accuracy", 0)
    classroom_accuracy = classroom_location.get("accuracy", 0)

    # Total accuracy margin (sum of both accuracies, with a minimum of 10m)
    accuracy_margin = max(student_accuracy + classroom_accuracy, 10)

    # Adjust max distance based on accuracy
    adjusted_max_distance = max_distance + accuracy_margin

    # Check if student is within the adjusted range
    is_within_range = distance <= adjusted_max_distance

    if is_within_range:
        logger.info(f"GPS verification passed: Student is {distance:.1f}m from classroom (allowed: {adjusted_max_distance:.1f}m)")
    else:
        logger.warning(f"GPS verification failed: Student is {distance:.1f}m from classroom (allowed: {adjusted_max_distance:.1f}m)")

    return is_within_range, distance

# Flask Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id'].strip()
        password = request.form['password']
        user = users_collection.find_one({"_id": user_id})
        if user and user.get("password") == hash_password(password):
            session['user_id'] = user['_id']
            session['role'] = user['role']
            session['name'] = user['name']
            logger.info(f"User {user_id} logged in as {user['role']}")
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('student_dashboard'))
        flash('Invalid credentials!')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    logger.info("User logged out")
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    total_students = users_collection.count_documents({"role": "student"})
    total_classes = classes_collection.count_documents({})
    total_schedules = schedules_collection.count_documents({})
    return render_template('admin_dashboard.html', name=session['name'],
                         total_students=total_students, total_classes=total_classes,
                         total_schedules=total_schedules)

@app.route('/admin/generate_class_qr', methods=['GET', 'POST'])
def generate_class_qr():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    classes = list(classes_collection.find({}))
    qr_code_url = None
    session_name = None
    class_code = None
    valid_until = None

    if request.method == 'POST':
        class_code = request.form['class_code']
        session_name = request.form['session_name']
        valid_minutes = int(request.form['valid_minutes'])

        # Get class details
        class_data = classes_collection.find_one({"code": class_code})
        if not class_data:
            flash("Class not found!")
            return redirect(url_for('generate_class_qr'))

        # Create session data
        now = datetime.now()
        valid_until = now + timedelta(minutes=valid_minutes)

        session_data = {
            "class_code": class_code,
            "class_name": class_data["name"],
            "session_name": session_name,
            "created_by": session['user_id'],
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "valid_until": valid_until.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Generate a unique session ID
        session_id = f"{class_code}_{int(now.timestamp())}"

        # Store in database
        db["class_sessions"].insert_one({"_id": session_id, **session_data})

        # Generate QR code with the session ID
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(session_id)
        qr.make(fit=True)
        qr_img = qr.make_image(fill='black', back_color='white')

        # Save QR code
        qr_path = os.path.join(app.config['UPLOAD_FOLDER'], 'qr_codes', f"class_{session_id}.png")
        qr_img.save(qr_path)

        # URL for the QR code
        qr_code_url = url_for('static', filename=f"uploads/qr_codes/class_{session_id}.png")
        valid_until = valid_until.strftime("%Y-%m-%d %H:%M:%S")

        flash(f"QR code generated for {class_data['name']} - {session_name}")

    return render_template('generate_class_qr.html', classes=classes,
                          qr_code_url=qr_code_url, session_name=session_name,
                          class_code=class_code, valid_until=valid_until)

@app.route('/student/dashboard')
def student_dashboard():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    enrollments = list(enrollments_collection.find({"student_id": session['user_id']}))
    class_codes = [e["class_code"] for e in enrollments]
    upcoming_schedules = list(schedules_collection.find({
        "class_code": {"$in": class_codes},
        "start_time": {"$gte": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    }).sort("start_time", 1).limit(5))
    return render_template('student_dashboard.html', name=session['name'], schedules=upcoming_schedules)

@app.route('/admin/register', methods=['GET', 'POST'])
def register():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    if request.method == 'POST':
        data = {
            "_id": request.form['reg_id'].strip(),
            "name": request.form['reg_name'].strip(),
            "role": request.form['reg_role'],
            "password": hash_password(request.form['reg_pass'])
        }
        if request.form['reg_role'] == 'student':
            section = request.form['reg_section'].strip()
            if not section:
                flash("Section is required for students!")
                return redirect(url_for('register'))
            data["section"] = section

        if not all(data.values()):
            flash("All fields are required!")
            return redirect(url_for('register'))
        if users_collection.find_one({"_id": data["_id"]}):
            flash("User ID already exists!")
            return redirect(url_for('register'))
        if data["role"] == "student":
            # Store user data in session temporarily
            session['temp_user_data'] = data
            # Redirect to face capture page
            return redirect(url_for('start_face_capture', user_id=data["_id"]))
        else:
            # For non-student users, complete registration immediately
            users_collection.insert_one(data)
            flash(f"User {data['name']} registered!")
            logger.info(f"User {data['_id']} registered as {data['role']}")
            return redirect(url_for('admin_dashboard'))
    return render_template('register.html')

@app.route('/admin/create_class', methods=['GET', 'POST'])
def create_class():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    if request.method == 'POST':
        class_data = {
            "name": request.form['class_name'].strip(),
            "code": request.form['class_code'].strip(),
            "description": request.form['class_desc'].strip(),
            "teacher_id": session['user_id'],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add GPS coordinates if provided
        if 'classroom_lat' in request.form and 'classroom_lon' in request.form:
            try:
                lat = float(request.form['classroom_lat'])
                lon = float(request.form['classroom_lon'])
                accuracy = float(request.form['location_accuracy']) if 'location_accuracy' in request.form else None

                class_data["location"] = {
                    "latitude": lat,
                    "longitude": lon,
                    "accuracy": accuracy,
                    "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                logger.info(f"Classroom location captured: {lat}, {lon} (accuracy: {accuracy}m)")
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing location data: {e}")
                flash("Warning: Could not process location data. Attendance location verification will be disabled.")
        if not class_data["name"] or not class_data["code"]:
            flash("Class Name and Class Code are required!")
            return redirect(url_for('create_class'))
        if classes_collection.find_one({"code": class_data["code"]}):
            flash("Class Code already exists!")
            return redirect(url_for('create_class'))
        classes_collection.insert_one(class_data)
        flash(f"Class {class_data['name']} created!")
        logger.info(f"Class {class_data['code']} created")
        return redirect(url_for('admin_dashboard'))
    return render_template('create_class.html')

@app.route('/admin/schedule_class', methods=['GET', 'POST'])
def schedule_class():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    classes = list(classes_collection.find({}))
    if request.method == 'POST':
        try:
            start_time_str = request.form['sched_start_time'].strip()
            end_time_str = request.form['sched_end_time'].strip()
            class_code = request.form['sched_class_code'].strip()
            start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M")
            if end_time <= start_time:
                flash("End time must be after start time!")
                return redirect(url_for('schedule_class'))
            class_data = classes_collection.find_one({"code": class_code})
            if not class_data:
                flash("Class code does not exist!")
                return redirect(url_for('schedule_class'))
            schedule_data = {
                "class_code": class_code,
                "class_name": class_data["name"],
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "created_by": session['user_id'],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            schedules_collection.insert_one(schedule_data)
            flash(f"Class {class_data['name']} scheduled!")
            logger.info(f"Class {class_code} scheduled from {start_time} to {end_time}")
            return redirect(url_for('admin_dashboard'))
        except ValueError:
            flash("Invalid date/time format!")
            return redirect(url_for('schedule_class'))
    return render_template('schedule_class.html', classes=classes)

@app.route('/admin/enroll_students', methods=['GET', 'POST'])
def enroll_students():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    classes = list(classes_collection.find({}))
    sections = users_collection.distinct("section", {"role": "student"})
    if request.method == 'POST':
        class_code = request.form['enroll_class_code'].strip()
        section = request.form['enroll_section'].strip()
        if not class_code or not section:
            flash("Class code and section are required!")
            return redirect(url_for('enroll_students'))
        if not classes_collection.find_one({"code": class_code}):
            flash("Class code does not exist!")
            return redirect(url_for('enroll_students'))
        students = users_collection.find({"role": "student", "section": section})
        student_ids = [student["_id"] for student in students]
        if not student_ids:
            flash(f"No students found in section {section}!")
            return redirect(url_for('enroll_students'))
        enrolled_count = 0
        for sid in student_ids:
            result = enrollments_collection.update_one(
                {"student_id": sid, "class_code": class_code},
                {"$set": {"student_id": sid, "class_code": class_code, "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}},
                upsert=True
            )
            if result.matched_count or result.upserted_id:
                enrolled_count += 1
        flash(f"Enrolled {enrolled_count} students from section {section} in class {class_code}!")
        logger.info(f"Enrolled {enrolled_count} students in class {class_code}")
        return redirect(url_for('admin_dashboard'))
    return render_template('enroll_students.html', classes=classes, sections=sections)

@app.route('/student/mark_attendance', methods=['GET'])
def mark_attendance():
    if 'user_id' not in session or session['role'] != 'student':
        logger.warning("Unauthorized access to mark_attendance")
        return redirect(url_for('login'))
    enrollments = enrollments_collection.find({"student_id": session['user_id']})
    class_codes = [e["class_code"] for e in enrollments]
    active_schedules = list(schedules_collection.find({
        "class_code": {"$in": class_codes},
        "start_time": {"$lte": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "end_time": {"$gte": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    }))
    logger.info(f"Rendering mark_attendance for user {session['user_id']} with {len(active_schedules)} active schedules")
    return render_template('mark_attendance.html', schedules=active_schedules)

@app.route('/student/view_attendance')
def view_student_attendance():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    records = list(attendance_collection.find({"student_id": session['user_id']}).sort("date", -1))
    return render_template('view_student_attendance.html', attendance=records)

@app.route('/student/view_schedule')
def view_class_schedule():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    enrollments = enrollments_collection.find({"student_id": session['user_id']})
    class_codes = [e["class_code"] for e in enrollments]
    schedule = list(schedules_collection.find({"class_code": {"$in": class_codes}}).sort("start_time", 1))
    for sched in schedule:
        sched["class_name"] = classes_collection.find_one({"code": sched["class_code"]})["name"]
    return render_template('view_class_schedule.html', schedule=schedule)

@app.route('/admin/view_reports', methods=['GET', 'POST'])
def view_attendance_reports():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    if request.method == 'POST':
        report_type = request.form['report_type']
        report_id = request.form['report_id'].strip()
        if report_type == "By Student":
            query = {"student_id": report_id} if report_id else {}
            records = list(attendance_collection.find(query).sort("date", -1))
            return render_template('view_attendance_reports.html', report_type=report_type, report_data=records)
        elif report_type == "By Class":
            query = {"class_code": report_id} if report_id else {}
            records = attendance_collection.find(query)
            class_totals = {}
            for record in records:
                class_code = record['class_code']
                class_name = record.get("class_name", classes_collection.find_one({"code": class_code})["name"])
                class_key = f"{class_name} ({class_code})"
                class_totals[class_key] = class_totals.get(class_key, {"present": 0, "total": 0})
                class_totals[class_key]["total"] += 1
                if record["status"] == "Present":
                    class_totals[class_key]["present"] += 1
            return render_template('view_attendance_reports.html', report_type=report_type, report_data=class_totals)
        elif report_type == "By Section":
            students = users_collection.find({"section": report_id, "role": "student"})
            student_ids = [s["_id"] for s in students]
            records = list(attendance_collection.find({"student_id": {"$in": student_ids}}).sort("date", -1))
            return render_template('view_attendance_reports.html', report_type=report_type, report_data=records)
    sections = users_collection.distinct("section", {"role": "student"})
    classes = list(classes_collection.find({}))
    return render_template('view_attendance_reports.html', sections=sections, classes=classes)

@app.route('/admin/manage_classes', methods=['GET', 'POST'])
def manage_classes():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    if request.method == 'POST':
        class_code = request.form['class_code'].strip()
        action = request.form['action']
        if action == "delete":
            classes_collection.delete_one({"code": class_code})
            schedules_collection.delete_many({"class_code": class_code})
            enrollments_collection.delete_many({"class_code": class_code})
            attendance_collection.delete_many({"class_code": class_code})
            flash(f"Class {class_code} and related data deleted!")
            logger.info(f"Class {class_code} deleted")
        return redirect(url_for('manage_classes'))
    classes = list(classes_collection.find({}))
    return render_template('manage_classes.html', classes=classes)

# SocketIO handlers
# SocketIO handlers (replace this section in your existing app.py)
@socketio.on('connect', namespace='/video_feed')
def handle_connect():
    logger.info("Client connected to video feed")

@socketio.on('disconnect', namespace='/video_feed')
def handle_disconnect():
    logger.info("Client disconnected from video feed")

@socketio.on('message', namespace='/video_feed')
def handle_video_frame(data):
    # Check if user is authenticated and is a student
    if 'user_id' not in session:
        logger.warning("Unauthorized WebSocket access - no user_id in session")
        emit('message', "Error: You are not logged in. Please log in first.")
        return

    if session['role'] != 'student':
        logger.warning(f"Unauthorized WebSocket access - user {session['user_id']} is not a student")
        emit('message', "Error: Only students can mark attendance")
        return

    # Handle structured data format
    if isinstance(data, dict):
        # Extract data from the message
        image_data = data.get('image')
        qr_detected = data.get('qrDetected', False)
        class_session_data = data.get('classSessionData')
        face_verified = data.get('faceVerified', False)
        location_verified = data.get('locationVerified', False)
        skip_image_verification = data.get('skipImageVerification', False)
        student_location_data = data.get('studentLocation')

        # Special case: If face is already verified and we have location data, proceed with location verification
        # even if there's no image data
        if face_verified and not location_verified and class_session_data and student_location_data:
            logger.info("Face already verified, proceeding with location verification")
            process_location_verification(class_session_data, student_location_data, student_id=session['user_id'])
            return

        # Check if we're explicitly skipping image verification
        if skip_image_verification and face_verified and class_session_data:
            logger.info("Skipping image verification for location check")
            if student_location_data:
                process_location_verification(class_session_data, student_location_data, student_id=session['user_id'])
                return
            else:
                logger.warning("No location data provided with skipImageVerification flag")
                emit('message', "Error: Location data not available. Please enable location services.")
                return

        # Check if this is a direct request to mark attendance
        if data.get('markAttendance') and class_session_data and face_verified and location_verified:
            # This is a direct request to mark attendance after all verifications are complete
            mark_attendance(class_session_data, student_id=session['user_id'])
            return

        # For normal processing (QR detection and face verification), require image data
        if not image_data:
            logger.warning("No image data received")
            emit('message', "Error: No image data received")
            return

        # Decode the image frame
        try:
            logger.debug("Attempting to decode frame")
            img_data = base64.b64decode(image_data.split(',')[1])
            img = Image.open(BytesIO(img_data))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            logger.debug("Frame decoded successfully")
        except Exception as e:
            logger.error("Error decoding frame: %s", e)
            emit('message', "Error: Failed to process video frame")
            return

        # This section is now handled earlier in the code

        # If we already detected a class QR code, proceed with verification steps
        if qr_detected and class_session_data:
            # Get student location from the data
            student_location_data = data.get('studentLocation')

            # Process verification steps in sequence
            # Only proceed to the next step if the previous step is completed
            if not face_verified:
                # First, verify face
                process_face_verification(frame, class_session_data, student_id=session['user_id'])
            elif face_verified and not location_verified and student_location_data:
                # Then, verify location (only if location data is available)
                process_location_verification(class_session_data, student_location_data, student_id=session['user_id'])
            return

        # Otherwise, scan for a class QR code
        qr_data = scan_qr(frame)
        if not qr_data:
            # Don't emit an error for every frame without a QR code
            return

        # Check if this is a class session QR code
        try:
            # Class session QR codes are in the format: class_code_timestamp
            if '_' in qr_data:
                session_id = qr_data.strip()
                class_session = db["class_sessions"].find_one({"_id": session_id})

                if class_session:
                    # Check if the session is still valid
                    now = datetime.now()
                    valid_until = datetime.strptime(class_session["valid_until"], "%Y-%m-%d %H:%M:%S")

                    if now > valid_until:
                        logger.warning(f"Class session QR code expired: {session_id}")
                        emit('message', "Error: This class QR code has expired")
                        return

                    # Check if student is enrolled in this class
                    student_id = session['user_id']
                    enrollment = enrollments_collection.find_one({
                        "student_id": student_id,
                        "class_code": class_session["class_code"]
                    })

                    if not enrollment:
                        logger.warning(f"Student {student_id} not enrolled in class {class_session['class_code']}")
                        emit('message', "Error: You are not enrolled in this class")
                        return

                    # Send the class session data back to the client
                    emit('message', {
                        "type": "qr_detected",
                        "data": class_session
                    })
                    return

            # If we get here, it's not a valid class session QR code
            logger.warning(f"Invalid QR code scanned: {qr_data}")
            emit('message', "Error: Invalid class QR code")

        except Exception as e:
            logger.error(f"Error processing QR code: {e}")
            emit('message', "Error: Failed to process QR code")
    else:
        # Legacy format - string data
        logger.warning("Received legacy format data")
        emit('message', "Error: Outdated client. Please refresh the page.")

def process_attendance_with_class_qr(frame, class_session_data, face_verified=False, location_verified=False, student_location=None):
    student_id = session['user_id']
    class_code = class_session_data.get('class_code')
    class_name = class_session_data.get('class_name')
    session_name = class_session_data.get('session_name')

    logger.info(f"Processing attendance for student {student_id} in class {class_code} - {session_name}")

    # Check if attendance is already marked
    existing_attendance = attendance_collection.find_one({
        "student_id": student_id,
        "class_code": class_code,
        "session_name": session_name
    })

    if existing_attendance:
        logger.info(f"Attendance already marked for {student_id} in {class_code} - {session_name}")
        emit('message', "Error: Attendance already marked for this class session")
        return

    # Get student data for face verification
    student = users_collection.find_one({"_id": student_id, "role": "student"})
    if not student:
        logger.warning(f"Student data not found: {student_id}")
        emit('message', "Error: Student data not found")
        return

    # If face is not yet verified, verify it
    if not face_verified:
        logger.debug(f"Attempting face verification for student {student_id}")
        if "face_data" not in student or not student["face_data"]:
            logger.warning(f"No face data found for student {student_id}")
            emit('message', "Error: No face data found for this student. Please contact administrator.")
            return

        verified, similarity_score = verify_face(student["face_data"], frame)
        if verified:
            # Face verified, send success message with similarity score
            logger.info(f"Face verification successful for {student_id} with score {similarity_score}")
            emit('message', {
                "type": "face_verified",
                "similarity": similarity_score
            })
            return
        elif similarity_score > 0.0:
            # Face detected but not verified, send similarity score for UI feedback
            logger.debug(f"Face verification in progress for {student_id} with score {similarity_score}")

            # If similarity is very low after multiple attempts, provide more detailed feedback
            # Use a unique identifier for tracking attempts (student_id is unique per session)
            if similarity_score < 0.5 and student_id in verification_attempts:
                verification_attempts[student_id] += 1
                if verification_attempts[student_id] > 10:  # After 10 attempts with low similarity
                    logger.warning(f"Persistent low similarity for {student_id}: {similarity_score}")
                    emit('message', "Error: Face verification failed. Please ensure your face is fully visible, well-lit, and not covered.")
                    verification_attempts[student_id] = 0  # Reset counter
                    return
            else:
                verification_attempts[student_id] = 1

            # Send progress update
            emit('message', {
                "type": "face_progress",
                "similarity": similarity_score
            })
            return
        else:
            # No face detected or features extracted
            logger.debug(f"Face verification still in progress for {student_id}")

            # After multiple failed attempts with no face detected, provide detailed feedback
            if student_id in detection_attempts:
                detection_attempts[student_id] += 1
                if detection_attempts[student_id] > 5:  # After 5 attempts with no face detected
                    logger.warning(f"Persistent face detection failure for {student_id}")
                    emit('message', "Error: No face detected. Please ensure you are looking directly at the camera, remove any face coverings, and ensure good lighting.")
                    detection_attempts[student_id] = 0  # Reset counter
                    # Don't return - continue processing frames until timeout
            else:
                detection_attempts[student_id] = 1

            # Don't proceed to location verification if face is not verified
            return

    # If location is not yet verified, verify it
    if face_verified and not location_verified:
        # Use the student_location parameter passed to the function
        # Verify GPS location
        is_verified, distance = verify_gps(class_code, student_location)

        if is_verified:
            # Location verified, send success message
            logger.info(f"Location verification successful for {student_id}")
            emit('message', {
                "type": "location_verified",
                "distance": distance
            })
            return
        else:
            # Format distance message if available
            distance_msg = ""
            if distance is not None:
                # Get the class data to check accuracy information
                class_data = classes_collection.find_one({"code": class_code})
                classroom_location = class_data.get("location", {})

                # Get accuracy information
                student_accuracy = student_location.get("accuracy", 0)
                classroom_accuracy = classroom_location.get("accuracy", 0)
                total_accuracy = max(student_accuracy + classroom_accuracy, 10)

                # Calculate the adjusted max distance
                base_distance = 50  # Base allowed distance
                adjusted_max = base_distance + total_accuracy

                distance_msg = f" You are {distance:.1f}m away from the classroom (allowed: {adjusted_max:.1f}m)."

            logger.warning(f"GPS verification failed for {student_id}")
            emit('message', f"Error: GPS verification failed! You must be in the classroom.{distance_msg}")
            return

    # If both face and location are verified, mark attendance
    if face_verified and location_verified:
        # Mark attendance
        now = datetime.now()
        date = now.strftime("%Y-%m-%d %H:%M:%S")

        attendance_data = {
            "student_id": student_id,
            "class_code": class_code,
            "class_name": class_name,
            "session_name": session_name,
            "date": date,
            "status": "Present",
            "gps_verified": True
        }

        try:
            attendance_collection.insert_one(attendance_data)
            logger.info(f"Attendance marked for {student_id} in {class_name} - {session_name} at {date}")
            emit('message', "Attendance Marked")
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            emit('message', "Error: Failed to mark attendance. Please try again.")

@app.route('/start_face_capture/<user_id>')
def start_face_capture(user_id):
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    if 'temp_user_data' not in session:
        flash("Registration data not found!")
        return redirect(url_for('register'))
    return render_template('face_capture.html', user_id=user_id)

@app.route('/capture_face_image', methods=['POST'])
def capture_face_image():
    if 'user_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    data = request.json
    user_id = data.get('user_id')
    image_data = data.get('image')
    count = data.get('count', 0)

    if not user_id or not image_data:
        return jsonify({'success': False, 'message': 'Missing data'}), 400

    try:
        # Create user folder if it doesn't exist
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'face_data', f"user_{user_id}")
        os.makedirs(user_folder, exist_ok=True)

        # Save the image
        image_data = image_data.split(',')[1]  # Remove the data URL prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # Convert to OpenCV format for face detection
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'}), 400

        # Save the image
        image_path = os.path.join(user_folder, f"{count}.jpg")
        cv2.imwrite(image_path, opencv_image)

        return jsonify({'success': True, 'message': 'Image captured successfully'})
    except Exception as e:
        logger.error(f"Error capturing face image: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/complete_face_registration', methods=['POST'])
def complete_face_registration():
    if 'user_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    if 'temp_user_data' not in session:
        return jsonify({'success': False, 'message': 'Registration data not found'}), 400

    data = request.json
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({'success': False, 'message': 'Missing user ID'}), 400

    try:
        # Get the user data from session
        user_data = session.pop('temp_user_data')

        # Get all face images
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'face_data', f"user_{user_id}")
        face_paths = [os.path.join(user_folder, f) for f in os.listdir(user_folder) if f.endswith('.jpg')]

        if not face_paths:
            return jsonify({'success': False, 'message': 'No face images captured'}), 400

        # Add face data to user data
        user_data['face_data'] = face_paths

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(user_id)
        qr.make(fit=True)
        qr_img = qr.make_image(fill='black', back_color='white')
        qr_path = os.path.join(app.config['UPLOAD_FOLDER'], 'qr_codes', f"qr_{user_id}.png")
        qr_img.save(qr_path)

        # Save user to database
        users_collection.insert_one(user_data)

        flash(f"User {user_data['name']} registered with face data!")
        logger.info(f"User {user_id} registered as {user_data['role']} with {len(face_paths)} face images")

        return jsonify({'success': True, 'redirect_url': url_for('admin_dashboard')})
    except Exception as e:
        logger.error(f"Error completing face registration: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# New verification functions
def process_face_verification(frame, class_session_data, student_id):
    """Handle face verification step"""
    class_code = class_session_data.get('class_code')
    class_name = class_session_data.get('class_name')
    session_name = class_session_data.get('session_name')

    logger.debug(f"Processing face verification for student {student_id} in class {class_code}")

    # Check if attendance is already marked
    existing_attendance = attendance_collection.find_one({
        "student_id": student_id,
        "class_code": class_code,
        "session_name": session_name
    })

    if existing_attendance:
        logger.info(f"Attendance already marked for {student_id} in {class_code} - {session_name}")

        # Get the date when attendance was marked
        marked_date = existing_attendance.get('date', 'unknown date')

        # Send a more detailed message with class name - use 'already marked' as a keyword for client-side detection
        emit('message', f"Error: Attendance already marked for {class_name} - {session_name} on {marked_date}. You don't need to mark attendance again.")
        return

    # Get student data for face verification
    student = users_collection.find_one({"_id": student_id, "role": "student"})
    if not student:
        logger.warning(f"Student data not found: {student_id}")
        emit('message', "Error: Student data not found")
        return

    # Verify face
    if "face_data" not in student or not student["face_data"]:
        logger.warning(f"No face data found for student {student_id}")
        emit('message', "Error: No face data found for this student. Please contact administrator.")
        return

    verified, similarity_score = verify_face(student["face_data"], frame)
    if verified:
        # Face verified, send success message with similarity score
        logger.info(f"Face verification successful for {student_id} with score {similarity_score}")
        emit('message', {
            "type": "face_verified",
            "similarity": similarity_score
        })
        return
    elif similarity_score > 0.0:
        # Face detected but not verified, send similarity score for UI feedback
        logger.debug(f"Face verification in progress for {student_id} with score {similarity_score}")

        # If similarity is very low after multiple attempts, provide more detailed feedback
        if similarity_score < 0.5 and student_id in verification_attempts:
            verification_attempts[student_id] += 1
            if verification_attempts[student_id] > 10:  # After 10 attempts with low similarity
                logger.warning(f"Persistent low similarity for {student_id}: {similarity_score}")
                emit('message', "Error: Face verification failed. Please ensure your face is fully visible, well-lit, and not covered.")
                verification_attempts[student_id] = 0  # Reset counter
                return
        else:
            verification_attempts[student_id] = 1

        # Send progress update
        emit('message', {
            "type": "face_progress",
            "similarity": similarity_score
        })
        return
    else:
        # No face detected or features extracted
        logger.debug(f"Face verification still in progress for {student_id}")

        # After multiple failed attempts with no face detected, provide detailed feedback
        if student_id in detection_attempts:
            detection_attempts[student_id] += 1
            if detection_attempts[student_id] > 5:  # After 5 attempts with no face detected
                logger.warning(f"Persistent face detection failure for {student_id}")
                emit('message', "Error: No face detected. Please ensure you are looking directly at the camera, remove any face coverings, and ensure good lighting.")
                detection_attempts[student_id] = 0  # Reset counter
                return
        else:
            detection_attempts[student_id] = 1
        return

def process_location_verification(class_session_data, student_location, student_id):
    """Handle location verification step"""
    class_code = class_session_data.get('class_code')

    logger.debug(f"Processing location verification for student {student_id} in class {class_code}")

    # Check if student location data is provided
    if not student_location:
        logger.warning(f"No location data provided for student {student_id}")
        emit('message', "Error: No location data available. Please ensure location services are enabled.")
        return

    # Verify GPS location
    is_verified, distance = verify_gps(class_code, student_location)

    if is_verified:
        # Location verified, send success message
        logger.info(f"Location verification successful for {student_id}")
        emit('message', {
            "type": "location_verified",
            "distance": distance
        })
        return
    else:
        # Format distance message if available
        distance_msg = ""
        if distance is not None:
            # Get the class data to check accuracy information
            class_data = classes_collection.find_one({"code": class_code})
            classroom_location = class_data.get("location", {})

            # Get accuracy information
            student_accuracy = student_location.get("accuracy", 0)
            classroom_accuracy = classroom_location.get("accuracy", 0)
            total_accuracy = max(student_accuracy + classroom_accuracy, 10)

            # Calculate the adjusted max distance
            base_distance = 50  # Base allowed distance
            adjusted_max = base_distance + total_accuracy

            distance_msg = f" You are {distance:.1f}m away from the classroom (allowed: {adjusted_max:.1f}m)."

        logger.warning(f"GPS verification failed for {student_id}")
        emit('message', f"Error: GPS verification failed! You must be in the classroom.{distance_msg}")
        return

def mark_attendance(class_session_data, student_id):
    """Mark attendance after all verification steps are complete"""
    class_code = class_session_data.get('class_code')
    class_name = class_session_data.get('class_name')
    session_name = class_session_data.get('session_name')

    logger.info(f"Marking attendance for student {student_id} in class {class_code} - {session_name}")

    # Check if attendance is already marked (double-check)
    existing_attendance = attendance_collection.find_one({
        "student_id": student_id,
        "class_code": class_code,
        "session_name": session_name
    })

    if existing_attendance:
        logger.info(f"Attendance already marked for {student_id} in {class_code} - {session_name}")

        # Get the date when attendance was marked
        marked_date = existing_attendance.get('date', 'unknown date')

        # Send a more detailed message - use 'already marked' as a keyword for client-side detection
        # Format the message to be clear and user-friendly
        emit('message', f"Error: Attendance already marked for {class_name} - {session_name} on {marked_date}. You don't need to mark attendance again.")
        return

    # Mark attendance
    now = datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M:%S")

    attendance_data = {
        "student_id": student_id,
        "class_code": class_code,
        "class_name": class_name,
        "session_name": session_name,
        "date": date,
        "status": "Present",
        "gps_verified": True
    }

    try:
        attendance_collection.insert_one(attendance_data)
        logger.info(f"Attendance marked for {student_id} in {class_name} - {session_name} at {date}")

        # Send a success message with completion flag
        emit('message', {
            "type": "attendance_marked",
            "message": "Attendance marked successfully!",
            "class_name": class_name,
            "session_name": session_name,
            "date": date
        })
    except Exception as e:
        logger.error(f"Error marking attendance: {e}")
        emit('message', "Error: Failed to mark attendance. Please try again.")

if __name__ == "__main__":
    if not users_collection.find_one({"role": "admin"}):
        users_collection.insert_one({
            "_id": "admin1",
            "name": "Admin User",
            "role": "admin",
            "password": hash_password("admin123")
        })
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)