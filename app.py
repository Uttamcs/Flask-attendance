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
import time
import base64
from io import BytesIO
from PIL import Image

# Try to import pyzbar, but provide a fallback if it's not available
try:
    from pyzbar.pyzbar import decode
    PYZBAR_AVAILABLE = True
    print("pyzbar library loaded successfully")
except ImportError as e:
    PYZBAR_AVAILABLE = False
    print(f"Warning: pyzbar not available - QR code scanning will be disabled: {e}")

# Load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Flask app setup
app = Flask(__name__)
# Use environment variable for secret key or generate a random one
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'qr_codes'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'face_data'), exist_ok=True)

# Set debug mode from environment variable
debug_mode = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

# SocketIO setup - use async mode for production
socketio = SocketIO(app, async_mode='eventlet' if not debug_mode else None)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Completely suppress Werkzeug logs
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.disabled = True

# Suppress Flask-SocketIO logs
engineio_logger = logging.getLogger('engineio')
engineio_logger.disabled = True
socketio_logger = logging.getLogger('socketio')
socketio_logger.disabled = True

# MongoDB setup
try:
    # Get MongoDB URI from environment variable or use default
    mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')

    # Extract database name from URI if present
    db_name = os.environ.get('MONGO_DB_NAME', 'syntra_attendance')

    # Handle Railway's MongoDB connection string format
    if '?retryWrites=' in mongo_uri and not db_name in mongo_uri:
        # Add database name to URI if not already present
        parts = mongo_uri.split('?')
        if len(parts) > 1:
            mongo_uri = f"{parts[0]}/{db_name}?{parts[1]}"

    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    logger.info("MongoDB connection successful.")
    client.server_info()

    # Get database from client
    if '/' in mongo_uri and not mongo_uri.endswith('/'):
        # If URI includes database name, use that database
        uri_parts = mongo_uri.split('/')
        if '?' in uri_parts[-1]:
            db_name = uri_parts[-1].split('?')[0]
        else:
            db_name = uri_parts[-1]

    db = client[db_name]
    logger.info(f"Using database: {db_name}")
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
    # Check if pyzbar is available
    if not PYZBAR_AVAILABLE:
        logger.warning("QR code scanning is disabled because pyzbar is not available")
        return None

    try:
        logger.debug("Attempting to scan QR code from frame")
        # Convert frame to grayscale for better QR detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try to decode from both the original and grayscale frames
        decoded = decode(frame) or decode(gray)

        if decoded:
            # Successfully decoded QR code
            qr_data = decoded[0].data.decode("utf-8").strip()
            logger.info(f"QR code detected: {qr_data}")
            return qr_data

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
            threshold = 0.80  # Set to 0.80 as requested

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
    max_distance = 20  # Increased from 20m to 20m

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
            # Store user information in session
            session['user_id'] = user['_id']
            session['role'] = user['role'].lower()  # Ensure lowercase for template conditionals
            session['name'] = user['name']

            # Add last login time
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
            session['last_login'] = current_time

            # Update last login in database
            users_collection.update_one(
                {"_id": user_id},
                {"$set": {"last_login": current_time}}
            )

            logger.info(f"User {user_id} logged in as {user['role']}")
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user['role'] == 'teacher':
                # Check if this teacher is assigned as a class advisor
                if user.get('section'):
                    session['is_advisor'] = True
                    session['advisor_section'] = user['section']
                else:
                    session['is_advisor'] = False
                return redirect(url_for('teacher_dashboard'))
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
    return render_template('admin_dashboard_new.html', name=session['name'],
                         total_students=total_students, total_classes=total_classes,
                         total_schedules=total_schedules, users_collection=users_collection)

@app.route('/admin/generate_class_qr', methods=['GET', 'POST'])
def generate_class_qr():
    if 'user_id' not in session or session['role'] not in ['admin', 'teacher']:
        return redirect(url_for('login'))

    # For teachers, show their own classes and classes they're assigned to
    if session['role'] == 'teacher':
        # Get classes created by this teacher
        own_classes = list(classes_collection.find({"teacher_id": session['user_id']}))

        # Get classes where this teacher is assigned to a schedule
        assigned_schedules = list(schedules_collection.find({"assigned_teacher_id": session['user_id']}))
        assigned_class_codes = [schedule["class_code"] for schedule in assigned_schedules]

        # Get the class details for assigned classes
        assigned_classes = list(classes_collection.find({"code": {"$in": assigned_class_codes}}))

        # Combine the lists, avoiding duplicates
        class_codes = set()
        classes = []

        for class_obj in own_classes + assigned_classes:
            if class_obj["code"] not in class_codes:
                class_codes.add(class_obj["code"])
                classes.append(class_obj)
    else:
        # For admins, show all classes
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

        # For teachers, verify they can only generate QR codes for classes they're assigned to
        if session['role'] == 'teacher':
            # Check if there's a current schedule for this class
            now = datetime.now()
            current_schedule = schedules_collection.find_one({
                "class_code": class_code,
                "start_time": {"$lte": now.strftime("%Y-%m-%d %H:%M:%S")},
                "end_time": {"$gte": now.strftime("%Y-%m-%d %H:%M:%S")}
            })

            if current_schedule:
                # Check if this teacher is assigned to this schedule
                if current_schedule.get('assigned_teacher_id') != session['user_id']:
                    flash("You are not the assigned teacher for this class schedule!")
                    return redirect(url_for('generate_class_qr'))
            else:
                # If no current schedule, check if they created the class
                if class_data.get('teacher_id') != session['user_id']:
                    flash("You can only generate QR codes for your own classes or classes you're assigned to!")
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
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # Higher error correction
            box_size=10,
            border=4
        )
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

    # Get student enrollments
    enrollments = list(enrollments_collection.find({"student_id": session['user_id']}))
    class_codes = [e["class_code"] for e in enrollments]

    # Get upcoming schedules
    upcoming_schedules = list(schedules_collection.find({
        "class_code": {"$in": class_codes},
        "start_time": {"$gte": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    }).sort("start_time", 1).limit(5))

    # Get attendance statistics
    attendance_records = list(attendance_collection.find({"student_id": session['user_id']}))
    total_sessions = len(attendance_records)
    present_sessions = sum(1 for record in attendance_records if record.get('status') == 'Present')

    # Calculate attendance percentage
    attendance_percentage = round((present_sessions / total_sessions * 100) if total_sessions > 0 else 0, 1)

    # Add current time for template to check if class is active
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get next class time
    next_class_time = upcoming_schedules[0]['start_time'] if upcoming_schedules else "No upcoming classes"

    return render_template('student_dashboard_new.html',
                          name=session['name'],
                          schedules=upcoming_schedules,
                          now=now,
                          total_classes=len(class_codes),
                          classes_attended=present_sessions,
                          attendance_percentage=attendance_percentage,
                          next_class_time=next_class_time)

@app.route('/teacher/dashboard')
def teacher_dashboard():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))

    # Get classes created by this teacher
    classes = list(classes_collection.find({"teacher_id": session['user_id']}))

    # Get class codes for classes created by this teacher
    class_codes = [c["code"] for c in classes]

    # Get schedules for these classes
    schedules = list(schedules_collection.find({"class_code": {"$in": class_codes}}).sort("start_time", -1))

    # Also get schedules where this teacher is assigned
    assigned_schedules = list(schedules_collection.find({"assigned_teacher_id": session['user_id']}).sort("start_time", -1))

    # Combine the schedules, avoiding duplicates
    schedule_ids = set()
    combined_schedules = []

    for schedule in schedules + assigned_schedules:
        # Create a unique ID for the schedule
        schedule_id = f"{schedule['class_code']}_{schedule['start_time']}_{schedule['end_time']}"
        if schedule_id not in schedule_ids:
            schedule_ids.add(schedule_id)
            combined_schedules.append(schedule)

    # Replace the schedules list with the combined list
    schedules = combined_schedules

    # Get recent attendance records for these classes
    recent_attendance = list(attendance_collection.find({"class_code": {"$in": class_codes}}).sort("date", -1).limit(10))

    # Add student names to attendance records
    for record in recent_attendance:
        student = users_collection.find_one({"_id": record["student_id"]})
        if student:
            record["student_name"] = student["name"]
        else:
            record["student_name"] = "Unknown Student"

    # Count total students enrolled in teacher's classes
    total_students = enrollments_collection.count_documents({"class_code": {"$in": class_codes}})

    return render_template('teacher_dashboard_new.html',
                          name=session['name'],
                          classes=classes,
                          total_classes=len(classes),
                          total_schedules=len(schedules),
                          total_students=total_students,
                          recent_attendance=recent_attendance)

@app.route('/advisor/dashboard')
def advisor_dashboard():
    if 'user_id' not in session or session['role'] != 'teacher' or not session.get('is_advisor', False):
        flash("You need to be a teacher assigned as a class advisor to access this page.")
        return redirect(url_for('login'))

    # Get advisor's section
    advisor = users_collection.find_one({"_id": session['user_id']})
    section = advisor.get("section", "Unknown")

    # Get students in this section
    students = list(users_collection.find({"role": "student", "section": section}))

    # Calculate attendance statistics for each student
    for student in students:
        attendance_records = list(attendance_collection.find({"student_id": student["_id"]}))
        total_sessions = len(attendance_records)
        present_sessions = sum(1 for record in attendance_records if record.get('status') == 'Present')
        student["attendance_rate"] = round((present_sessions / total_sessions * 100) if total_sessions > 0 else 0, 1)

    # Get recent attendance records for this section
    student_ids = [s["_id"] for s in students]
    recent_attendance = list(attendance_collection.find({"student_id": {"$in": student_ids}}).sort("date", -1).limit(10))

    # Add student names to attendance records
    for record in recent_attendance:
        student = next((s for s in students if s["_id"] == record["student_id"]), None)
        if student:
            record["student_name"] = student["name"]
        else:
            record["student_name"] = "Unknown Student"

    # Calculate overall attendance rate
    all_attendance = list(attendance_collection.find({"student_id": {"$in": student_ids}}))
    total_records = len(all_attendance)
    present_records = sum(1 for record in all_attendance if record.get('status') == 'Present')
    attendance_rate = round((present_records / total_records * 100) if total_records > 0 else 0, 1)

    # Get classes that students in this section are enrolled in
    enrolled_classes = enrollments_collection.distinct("class_code", {"student_id": {"$in": student_ids}})
    total_classes = len(enrolled_classes)

    # Get classes created by this advisor
    advisor_classes = list(classes_collection.find({"teacher_id": session['user_id']}))

    # Check which classes have students enrolled
    for cls in advisor_classes:
        # Count how many students from this section are enrolled in this class
        enrolled_count = enrollments_collection.count_documents({
            "class_code": cls["code"],
            "student_id": {"$in": student_ids}
        })
        cls["enrolled_count"] = enrolled_count
        cls["total_students"] = len(student_ids)
        cls["enrollment_percentage"] = round((enrolled_count / len(student_ids) * 100) if len(student_ids) > 0 else 0)

    return render_template('advisor_dashboard.html',
                          name=session['name'],
                          section=section,
                          students=students,
                          total_students=len(students),
                          total_classes=total_classes,
                          attendance_rate=attendance_rate,
                          recent_attendance=recent_attendance,
                          advisor_classes=advisor_classes)

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
        if request.form['reg_role'] in ['student', 'teacher']:
            section = request.form['reg_section'].strip()

            # Section is required for students
            if request.form['reg_role'] == 'student' and not section:
                flash("Section is required for students!")
                return redirect(url_for('register'))

            # If section is provided for a teacher, they become a class advisor
            if section:
                data["section"] = section

                # For teachers with section, check if this section already has an advisor
                if request.form['reg_role'] == 'teacher':
                    existing_advisor = users_collection.find_one({"role": "teacher", "section": section})
                    if existing_advisor:
                        flash(f"Section {section} already has a class advisor assigned: {existing_advisor['name']}!")
                        return redirect(url_for('register'))

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
    if 'user_id' not in session or session['role'] not in ['admin', 'teacher']:
        return redirect(url_for('login'))

    # For teachers, only allow access if they are class advisors
    if session['role'] == 'teacher' and not session.get('is_advisor', False):
        flash("You need to be a class advisor to create classes.")
        return redirect(url_for('teacher_dashboard'))

    if request.method == 'POST':
        class_data = {
            "name": request.form['class_name'].strip(),
            "code": request.form['class_code'].strip(),
            "description": request.form['class_desc'].strip(),
            "teacher_id": session['user_id'],
            "teacher_name": session['name'],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # If this is a class advisor, add the section to the class data
        if session['role'] == 'teacher' and session.get('is_advisor', False):
            class_data["section"] = session.get('advisor_section')

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
        logger.info(f"Class {class_data['code']} created by {session['role']} {session['user_id']}")

        # If the user is a class advisor, automatically enroll students from their section
        if session['role'] == 'teacher' and session.get('is_advisor', False):
            advisor_section = session.get('advisor_section')
            if advisor_section:
                # Get all students from the advisor's section
                students = users_collection.find({"role": "student", "section": advisor_section})
                student_ids = [student["_id"] for student in students]

                if student_ids:
                    enrolled_count = 0
                    for sid in student_ids:
                        result = enrollments_collection.update_one(
                            {"student_id": sid, "class_code": class_data["code"]},
                            {"$set": {"student_id": sid, "class_code": class_data["code"], "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}},
                            upsert=True
                        )
                        if result.matched_count or result.upserted_id:
                            enrolled_count += 1

                    if enrolled_count > 0:
                        flash(f"Automatically enrolled {enrolled_count} students from section {advisor_section} in this class!")
                        logger.info(f"Automatically enrolled {enrolled_count} students from section {advisor_section} in class {class_data['code']}")

        # Redirect based on user role
        if session['role'] == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif session.get('is_advisor', False):
            return redirect(url_for('advisor_manage_classes'))
        else:
            return redirect(url_for('teacher_dashboard'))

    return render_template('create_class.html')

@app.route('/admin/schedule_class', methods=['GET', 'POST'])
def schedule_class():
    if 'user_id' not in session or session['role'] not in ['admin', 'teacher']:
        return redirect(url_for('login'))

    # For teachers, only allow access if they are class advisors
    if session['role'] == 'teacher' and not session.get('is_advisor', False):
        flash("You need to be a class advisor to schedule classes.")
        return redirect(url_for('teacher_dashboard'))

    # For teachers, only show their own classes
    if session['role'] == 'teacher':
        classes = list(classes_collection.find({"teacher_id": session['user_id']}))
    else:
        # For admins, show all classes
        classes = list(classes_collection.find({}))

    # Get all teachers for the assignment dropdown
    teachers = list(users_collection.find({"role": "teacher"}))

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

            # Get the assigned teacher ID from the form
            assigned_teacher_id = request.form.get('assigned_teacher_id', session['user_id'])

            # Verify the assigned teacher exists and is a teacher
            assigned_teacher = users_collection.find_one({"_id": assigned_teacher_id, "role": "teacher"})
            if not assigned_teacher:
                flash("Invalid teacher assignment. Using yourself as the assigned teacher.")
                assigned_teacher_id = session['user_id']
                assigned_teacher = users_collection.find_one({"_id": assigned_teacher_id})

            schedule_data = {
                "class_code": class_code,
                "class_name": class_data["name"],
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "created_by": session['user_id'],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "assigned_teacher_id": assigned_teacher_id,
                "assigned_teacher_name": assigned_teacher["name"]
            }

            # For teachers, verify they can only schedule their own classes
            if session['role'] == 'teacher' and class_data.get('teacher_id') != session['user_id']:
                flash("You can only schedule your own classes!")
                return redirect(url_for('schedule_class'))

            schedules_collection.insert_one(schedule_data)
            flash(f"Class {class_data['name']} scheduled!")
            logger.info(f"Class {class_code} scheduled from {start_time} to {end_time} by {session['role']} {session['user_id']}")

            # If the user is a class advisor, automatically enroll students from their section
            if session['role'] == 'teacher' and session.get('is_advisor', False):
                advisor_section = session.get('advisor_section')
                if advisor_section:
                    # Get all students from the advisor's section
                    students = users_collection.find({"role": "student", "section": advisor_section})
                    student_ids = [student["_id"] for student in students]

                    if student_ids:
                        enrolled_count = 0
                        for sid in student_ids:
                            result = enrollments_collection.update_one(
                                {"student_id": sid, "class_code": class_code},
                                {"$set": {"student_id": sid, "class_code": class_code, "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}},
                                upsert=True
                            )
                            if result.matched_count or result.upserted_id:
                                enrolled_count += 1

                        if enrolled_count > 0:
                            flash(f"Automatically enrolled {enrolled_count} students from section {advisor_section} in this class!")
                            logger.info(f"Automatically enrolled {enrolled_count} students from section {advisor_section} in class {class_code}")

            # Redirect based on user role
            if session['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif session.get('is_advisor', False):
                return redirect(url_for('advisor_manage_classes'))
            else:
                return redirect(url_for('teacher_dashboard'))
        except ValueError:
            flash("Invalid date/time format!")
            return redirect(url_for('schedule_class'))

    return render_template('schedule_class.html', classes=classes, teachers=teachers)

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

    # Get student enrollments
    enrollments = enrollments_collection.find({"student_id": session['user_id']})
    class_codes = [e["class_code"] for e in enrollments]

    # Get active schedules
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_schedules = list(schedules_collection.find({
        "class_code": {"$in": class_codes},
        "start_time": {"$lte": now},
        "end_time": {"$gte": now}
    }))

    # Get student data for face verification
    student = users_collection.find_one({"_id": session['user_id']})
    has_face_data = "face_data" in student and student["face_data"]

    # Check if student has already marked attendance for any active classes
    marked_attendance = {}
    for schedule in active_schedules:
        class_code = schedule["class_code"]
        existing = attendance_collection.find_one({
            "student_id": session['user_id'],
            "class_code": class_code,
            "date": {"$gte": schedule["start_time"], "$lte": schedule["end_time"]}
        })
        marked_attendance[class_code] = True if existing else False

    logger.info(f"Rendering mark_attendance for user {session['user_id']} with {len(active_schedules)} active schedules")
    return render_template('mark_attendance_new.html',
                          schedules=active_schedules,
                          has_face_data=has_face_data,
                          marked_attendance=marked_attendance)

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
    if 'user_id' not in session or session['role'] not in ['admin', 'teacher']:
        return redirect(url_for('login'))
    if request.method == 'POST':
        report_type = request.form['report_type']
        report_id = request.form['report_id'].strip()
        if report_type == "By Student":
            query = {"student_id": report_id} if report_id else {}
            records = list(attendance_collection.find(query).sort("date", -1))

            # Enhance records with class names
            for record in records:
                class_code = record.get('class_code')
                if class_code and not record.get('class_name'):
                    class_data = classes_collection.find_one({"code": class_code})
                    if class_data:
                        record['class_name'] = class_data.get('name', 'Unknown Class')

            return render_template('view_attendance_reports.html', report_type=report_type, report_data=records)
        elif report_type == "By Class":
            query = {"class_code": report_id} if report_id else {}
            records = attendance_collection.find(query)
            class_totals = {}
            for record in records:
                class_code = record['class_code']
                # Get class name from the record or from the classes collection
                class_data = classes_collection.find_one({"code": class_code})
                if class_data:
                    class_name = record.get("class_name", class_data.get("name", "Unknown Class"))
                else:
                    class_name = record.get("class_name", "Unknown Class")

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

            # Enhance records with class names
            for record in records:
                class_code = record.get('class_code')
                if class_code and not record.get('class_name'):
                    class_data = classes_collection.find_one({"code": class_code})
                    if class_data:
                        record['class_name'] = class_data.get('name', 'Unknown Class')

            return render_template('view_attendance_reports.html', report_type=report_type, report_data=records)
    sections = users_collection.distinct("section", {"role": "student"})

    # For teachers, only show their own classes
    if session['role'] == 'teacher':
        classes = list(classes_collection.find({"teacher_id": session['user_id']}))
    else:
        # For admins, show all classes
        classes = list(classes_collection.find({}))

    return render_template('view_attendance_reports.html', sections=sections, classes=classes)

@app.route('/admin/manage_classes', methods=['GET', 'POST'])
def manage_classes():
    if 'user_id' not in session or session['role'] not in ['admin', 'teacher']:
        return redirect(url_for('login'))

    # For teachers, only allow access if they are class advisors
    if session['role'] == 'teacher' and not session.get('is_advisor', False):
        flash("You need to be a class advisor to manage classes.")
        return redirect(url_for('teacher_dashboard'))

    if request.method == 'POST':
        class_code = request.form['class_code'].strip()
        action = request.form['action']

        # For class advisors, only allow managing classes for their section
        if session['role'] == 'teacher' and session.get('is_advisor', False):
            # Get the class to check if it belongs to the advisor's section
            class_data = classes_collection.find_one({"code": class_code})
            if not class_data:
                flash("Class not found!")
                return redirect(url_for('manage_classes'))

            # Check if the class is associated with the advisor's section
            # This would require adding a section field to classes or checking enrollments
            # For now, we'll allow advisors to manage all classes they created
            if class_data.get('teacher_id') != session['user_id']:
                flash("You can only manage classes you created!")
                return redirect(url_for('manage_classes'))

        if action == "delete":
            classes_collection.delete_one({"code": class_code})
            schedules_collection.delete_many({"class_code": class_code})
            enrollments_collection.delete_many({"class_code": class_code})
            attendance_collection.delete_many({"class_code": class_code})
            flash(f"Class {class_code} and related data deleted!")
            logger.info(f"Class {class_code} deleted")

        # Redirect based on user role
        if session['role'] == 'admin':
            return redirect(url_for('manage_classes'))
        else:
            return redirect(url_for('advisor_manage_classes'))

    # Get classes based on user role
    if session['role'] == 'admin':
        classes = list(classes_collection.find({}))
    else:
        # For class advisors, only show classes they created
        classes = list(classes_collection.find({"teacher_id": session['user_id']}))

    return render_template('manage_classes.html', classes=classes)

@app.route('/admin/manage_sections', methods=['GET'])
def manage_sections():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    # Get all sections from all records (students, placeholders, etc.)
    sections_list = users_collection.distinct("section")
    sections = []

    for section_name in sections_list:
        if not section_name:  # Skip empty section names
            continue

        # Get advisor for this section (teacher with section assignment)
        advisor = users_collection.find_one({"role": "teacher", "section": section_name})

        # Count students in this section
        student_count = users_collection.count_documents({"role": "student", "section": section_name})

        sections.append({
            "name": section_name,
            "advisor": advisor,
            "student_count": student_count
        })

    # Get all teachers who are not already assigned as advisors
    advisors = list(users_collection.find({"role": "teacher", "section": {"$exists": False}}))

    # Get all teachers (for changing advisors)
    all_teachers = list(users_collection.find({"role": "teacher"}))

    return render_template('manage_sections.html',
                          sections=sections,
                          advisors=advisors,
                          all_teachers=all_teachers)

@app.route('/admin/add_section', methods=['POST'])
def add_section():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        section_name = request.form['section_name'].strip()

        if not section_name:
            flash("Section name is required!")
            return redirect(url_for('manage_sections'))

        # Check if section already exists
        if section_name in users_collection.distinct("section"):
            flash(f"Section {section_name} already exists!")
            return redirect(url_for('manage_sections'))

        # Create a dummy student to establish the section
        # This is just a placeholder and will be removed when real students are added
        placeholder_data = {
            "_id": f"placeholder_{section_name}_{int(datetime.now().timestamp())}",
            "name": f"Placeholder for section {section_name}",
            "role": "placeholder",
            "section": section_name,
            "password": hash_password("placeholder"),
            "is_placeholder": True
        }

        users_collection.insert_one(placeholder_data)
        flash(f"Section {section_name} created successfully!")

    return redirect(url_for('manage_sections'))

@app.route('/admin/assign_advisor', methods=['POST'])
def assign_advisor():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        section_id = request.form['section_id'].strip()
        advisor_id = request.form['advisor_id'].strip()
        change_advisor = request.form.get('change_advisor', 'false') == 'true'

        if not section_id or not advisor_id:
            flash("Section and advisor are required!")
            return redirect(url_for('manage_sections'))

        # Check if this section already has an advisor
        existing_advisor = users_collection.find_one({"role": "teacher", "section": section_id})

        if existing_advisor and not change_advisor:
            flash(f"Section {section_id} already has an advisor assigned: {existing_advisor['name']}. Please use the 'Change Advisor' option instead.")
            return redirect(url_for('manage_sections'))

        # Check if this teacher is already an advisor for another section
        teacher_info = users_collection.find_one({"_id": advisor_id})
        if teacher_info and teacher_info.get("section"):
            flash(f"This teacher is already assigned as an advisor for section {teacher_info['section']}.")
            return redirect(url_for('manage_sections'))

        # If we're changing an advisor, remove the section from the existing advisor
        if existing_advisor and change_advisor:
            users_collection.update_one(
                {"_id": existing_advisor["_id"]},
                {"$unset": {"section": ""}}
            )
            logger.info(f"Removed advisor {existing_advisor['_id']} from section {section_id}")

        # Update the teacher with the section assignment (making them a class advisor)
        result = users_collection.update_one(
            {"_id": advisor_id, "role": "teacher"},
            {"$set": {"section": section_id}}
        )

        if result.modified_count > 0:
            if change_advisor:
                flash(f"Advisor for section {section_id} changed successfully!")
            else:
                flash(f"Advisor assigned to section {section_id} successfully!")
        else:
            flash("Failed to assign advisor. Please try again.")

    return redirect(url_for('manage_sections'))

@app.route('/admin/remove_advisor_assignment')
def remove_advisor_assignment():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    advisor_id = request.args.get('advisor_id')

    if not advisor_id:
        flash("Advisor ID is required!")
        return redirect(url_for('manage_sections'))

    # Remove the section assignment from the teacher (removing class advisor role)
    users_collection.update_one(
        {"_id": advisor_id, "role": "teacher"},
        {"$unset": {"section": ""}}
    )

    flash("Advisor assignment removed successfully!")
    return redirect(url_for('manage_sections'))

@app.route('/admin/view_section_students')
def view_section_students():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    section = request.args.get('section')

    if not section:
        flash("Section is required!")
        return redirect(url_for('manage_sections'))

    # Get students in this section
    students = list(users_collection.find({"role": "student", "section": section}))

    # Get advisor for this section (teacher with section assignment)
    advisor = users_collection.find_one({"role": "teacher", "section": section})

    return render_template('section_students.html',
                          section=section,
                          students=students,
                          advisor=advisor)

@app.route('/advisor/section_attendance_report')
def section_attendance_report():
    if 'user_id' not in session or session['role'] != 'teacher' or not session.get('is_advisor', False):
        flash("You need to be a teacher assigned as a class advisor to access this page.")
        return redirect(url_for('login'))

    # Get advisor's section
    advisor = users_collection.find_one({"_id": session['user_id']})
    section = advisor.get("section", "Unknown")

    # Get students in this section
    students = list(users_collection.find({"role": "student", "section": section}))
    student_ids = [s["_id"] for s in students]

    # Get attendance records for these students
    attendance_records = list(attendance_collection.find({"student_id": {"$in": student_ids}}).sort("date", -1))

    # Enhance records with class names
    for record in attendance_records:
        class_code = record.get('class_code')
        if class_code and not record.get('class_name'):
            class_data = classes_collection.find_one({"code": class_code})
            if class_data:
                record['class_name'] = class_data.get('name', 'Unknown Class')

    # Group attendance by student
    student_attendance = {}
    for student in students:
        student_attendance[student["_id"]] = {
            "name": student["name"],
            "records": [],
            "present": 0,
            "total": 0,
            "percentage": 0
        }

    # Process attendance records
    for record in attendance_records:
        student_id = record["student_id"]
        if student_id in student_attendance:
            student_attendance[student_id]["records"].append(record)
            student_attendance[student_id]["total"] += 1
            if record.get("status") == "Present":
                student_attendance[student_id]["present"] += 1

    # Calculate percentages
    for student_id, data in student_attendance.items():
        if data["total"] > 0:
            data["percentage"] = round((data["present"] / data["total"]) * 100, 1)

    return render_template('section_attendance_report.html',
                          section=section,
                          students=students,
                          student_attendance=student_attendance)

@app.route('/advisor/manage_section_students')
def manage_section_students():
    if 'user_id' not in session or session['role'] != 'teacher' or not session.get('is_advisor', False):
        flash("You need to be a teacher assigned as a class advisor to access this page.")
        return redirect(url_for('login'))

    # Get advisor's section
    advisor = users_collection.find_one({"_id": session['user_id']})
    section = advisor.get("section", "Unknown")

    # Get students in this section
    students = list(users_collection.find({"role": "student", "section": section}))
    student_ids = [s["_id"] for s in students]

    # Get attendance records for these students
    attendance_records = list(attendance_collection.find({"student_id": {"$in": student_ids}}))

    # Calculate attendance rates for each student
    for student in students:
        student_id = student["_id"]
        student_records = [r for r in attendance_records if r["student_id"] == student_id]
        total_records = len(student_records)
        present_records = sum(1 for r in student_records if r.get("status") == "Present")

        # Calculate attendance rate
        attendance_rate = round((present_records / total_records * 100) if total_records > 0 else 0, 1)
        student["attendance_rate"] = attendance_rate
        student["total_classes"] = total_records
        student["present_classes"] = present_records

        # Get last login time
        student["last_login"] = student.get("last_login", "Never")

    # Calculate summary statistics
    perfect_count = sum(1 for s in students if s.get("attendance_rate", 0) == 100)
    good_count = sum(1 for s in students if s.get("attendance_rate", 0) >= 90 and s.get("attendance_rate", 0) < 100)
    avg_count = sum(1 for s in students if s.get("attendance_rate", 0) >= 75 and s.get("attendance_rate", 0) < 90)
    poor_count = sum(1 for s in students if s.get("attendance_rate", 0) < 75)

    attendance_summary = {
        "perfect_count": perfect_count,
        "good_count": good_count,
        "avg_count": avg_count,
        "poor_count": poor_count
    }

    return render_template('manage_section_students.html',
                          section=section,
                          students=students,
                          attendance_summary=attendance_summary)

@app.route('/advisor/manage_classes', methods=['GET', 'POST'])
def advisor_manage_classes():
    if 'user_id' not in session or session['role'] != 'teacher' or not session.get('is_advisor', False):
        flash("You need to be a teacher assigned as a class advisor to access this page.")
        return redirect(url_for('login'))

    # Get advisor's section
    advisor = users_collection.find_one({"_id": session['user_id']})
    section = advisor.get("section", "Unknown")

    if request.method == 'POST':
        class_code = request.form['class_code'].strip()
        action = request.form['action']

        # Get the class to check if it belongs to the advisor
        class_data = classes_collection.find_one({"code": class_code})
        if not class_data:
            flash("Class not found!")
            return redirect(url_for('advisor_manage_classes'))

        # Check if the class is created by this advisor
        if class_data.get('teacher_id') != session['user_id']:
            flash("You can only manage classes you created!")
            return redirect(url_for('advisor_manage_classes'))

        if action == "delete":
            classes_collection.delete_one({"code": class_code})
            schedules_collection.delete_many({"class_code": class_code})
            enrollments_collection.delete_many({"class_code": class_code})
            attendance_collection.delete_many({"class_code": class_code})
            flash(f"Class {class_code} and related data deleted!")
            logger.info(f"Class {class_code} deleted by advisor {session['user_id']}")

    # Get classes created by this advisor
    classes = list(classes_collection.find({"teacher_id": session['user_id']}))

    return render_template('advisor_manage_classes.html',
                          section=section,
                          classes=classes)

@app.route('/advisor/email_section_students')
def email_section_students():
    if 'user_id' not in session or session['role'] != 'teacher' or not session.get('is_advisor', False):
        flash("You need to be a teacher assigned as a class advisor to access this page.")
        return redirect(url_for('login'))

    # Get advisor's section
    advisor = users_collection.find_one({"_id": session['user_id']})
    section = advisor.get("section", "Unknown")

    # Get students in this section
    students = list(users_collection.find({"role": "student", "section": section}))

    # In a real application, you would send emails here
    # For now, we'll just simulate it
    student_count = len(students)
    flash(f"Email notification sent to {student_count} students in section {section}.")

    return redirect(url_for('manage_section_students'))

@app.route('/advisor/export_section_data')
def export_section_data():
    if 'user_id' not in session or session['role'] != 'teacher' or not session.get('is_advisor', False):
        flash("You need to be a teacher assigned as a class advisor to access this page.")
        return redirect(url_for('login'))

    # Get advisor's section
    advisor = users_collection.find_one({"_id": session['user_id']})
    section = advisor.get("section", "Unknown")

    # Get students in this section
    students = list(users_collection.find({"role": "student", "section": section}))

    # In a real application, you would generate a CSV or Excel file here
    # For now, we'll just simulate it
    student_count = len(students)
    flash(f"Data for {student_count} students in section {section} has been exported.")

    return redirect(url_for('manage_section_students'))

@app.route('/advisor/email_student')
def email_student():
    if 'user_id' not in session or session['role'] != 'teacher' or not session.get('is_advisor', False):
        flash("You need to be a teacher assigned as a class advisor to access this page.")
        return redirect(url_for('login'))

    student_id = request.args.get('student_id')
    if not student_id:
        flash("Student ID is required!")
        return redirect(url_for('manage_section_students'))

    # Get student details
    student = users_collection.find_one({"_id": student_id, "role": "student"})
    if not student:
        flash("Student not found!")
        return redirect(url_for('manage_section_students'))

    # In a real application, you would send an email here
    # For now, we'll just simulate it
    flash(f"Email notification sent to student {student['name']}.")

    return redirect(url_for('manage_section_students'))

@app.route('/advisor/flag_student')
def flag_student():
    if 'user_id' not in session or session['role'] != 'teacher' or not session.get('is_advisor', False):
        flash("You need to be a teacher assigned as a class advisor to access this page.")
        return redirect(url_for('login'))

    student_id = request.args.get('student_id')
    if not student_id:
        flash("Student ID is required!")
        return redirect(url_for('manage_section_students'))

    # Get student details
    student = users_collection.find_one({"_id": student_id, "role": "student"})
    if not student:
        flash("Student not found!")
        return redirect(url_for('manage_section_students'))

    # Toggle the flagged status
    current_flag = student.get("flagged", False)
    users_collection.update_one(
        {"_id": student_id},
        {"$set": {"flagged": not current_flag}}
    )

    action = "removed from" if current_flag else "added to"
    flash(f"Student {student['name']} {action} flagged list.")

    return redirect(url_for('manage_section_students'))

@app.route('/advisor/view_student_attendance_report')
def view_student_attendance_report():
    if 'user_id' not in session or session['role'] not in ['admin', 'teacher']:
        return redirect(url_for('login'))

    # Check if this teacher is assigned as a class advisor
    if session['role'] == 'teacher' and not session.get('is_advisor', False):
        flash("You need to be a teacher assigned as a class advisor to access this page.")
        return redirect(url_for('teacher_dashboard'))

    student_id = request.args.get('student_id')

    if not student_id:
        flash("Student ID is required!")
        if session['role'] == 'admin':
            return redirect(url_for('view_attendance_reports'))
        else:
            return redirect(url_for('advisor_dashboard'))

    # Get student details
    student = users_collection.find_one({"_id": student_id, "role": "student"})

    if not student:
        flash("Student not found!")
        if session['role'] == 'admin':
            return redirect(url_for('view_attendance_reports'))
        else:
            return redirect(url_for('advisor_dashboard'))

    # For teachers acting as class advisors, verify they can only view students in their section
    if session['role'] == 'teacher' and session.get('is_advisor', False):
        # Get the advisor's section from the database
        advisor = users_collection.find_one({"_id": session['user_id']})
        advisor_section = advisor.get("section", "Unknown")

        if student.get("section") != advisor_section:
            flash("You can only view students in your assigned section!")
            return redirect(url_for('advisor_dashboard'))

    # Get attendance records for this student
    attendance_records = list(attendance_collection.find({"student_id": student_id}).sort("date", -1))

    # Enhance records with class names
    for record in attendance_records:
        class_code = record.get('class_code')
        if class_code and not record.get('class_name'):
            class_data = classes_collection.find_one({"code": class_code})
            if class_data:
                record['class_name'] = class_data.get('name', 'Unknown Class')

    # Group by class
    class_attendance = {}
    for record in attendance_records:
        class_code = record.get("class_code")
        if class_code not in class_attendance:
            # Get class name from the record or from the classes collection
            class_data = classes_collection.find_one({"code": class_code})
            if class_data:
                class_name = record.get("class_name", class_data.get("name", "Unknown Class"))
            else:
                class_name = record.get("class_name", "Unknown Class")

            class_attendance[class_code] = {
                "name": class_name,
                "records": [],
                "present": 0,
                "total": 0,
                "percentage": 0
            }

        class_attendance[class_code]["records"].append(record)
        class_attendance[class_code]["total"] += 1
        if record.get("status") == "Present":
            class_attendance[class_code]["present"] += 1

    # Calculate percentages
    for class_code, data in class_attendance.items():
        if data["total"] > 0:
            data["percentage"] = round((data["present"] / data["total"]) * 100, 1)

    # Calculate overall attendance
    total_records = len(attendance_records)
    present_records = sum(1 for record in attendance_records if record.get("status") == "Present")
    overall_percentage = round((present_records / total_records * 100) if total_records > 0 else 0, 1)

    return render_template('student_attendance_report.html',
                          student=student,
                          class_attendance=class_attendance,
                          attendance_records=attendance_records,
                          total_records=total_records,
                          present_records=present_records,
                          overall_percentage=overall_percentage)

# User Profile and Settings Routes
@app.route('/profile')
def user_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get user data from database
    user = users_collection.find_one({"_id": session['user_id']})
    if not user:
        session.clear()
        return redirect(url_for('login'))

    # Get attendance statistics for students
    attendance_stats = None
    if session['role'] == 'student':
        # Get total classes the student is enrolled in
        enrollments = list(enrollments_collection.find({"student_id": session['user_id']}))
        total_classes = len(enrollments)

        # Get attendance records
        attendance_records = list(attendance_collection.find({"student_id": session['user_id']}))
        total_sessions = len(attendance_records)
        present_sessions = sum(1 for record in attendance_records if record.get('status') == 'Present')

        # Calculate attendance percentage
        attendance_percentage = (present_sessions / total_sessions * 100) if total_sessions > 0 else 0

        attendance_stats = {
            'total_classes': total_classes,
            'total_sessions': total_sessions,
            'present_sessions': present_sessions,
            'attendance_percentage': round(attendance_percentage, 2)
        }

    # For teachers/admins, get different stats
    admin_stats = None
    if session['role'] in ['admin', 'teacher']:
        # Get classes created/managed by this admin/teacher
        classes_managed = list(classes_collection.find({"created_by": session['user_id']}))
        total_managed = len(classes_managed)

        # Get total students if admin
        total_students = users_collection.count_documents({"role": "student"}) if session['role'] == 'admin' else None

        admin_stats = {
            'total_managed': total_managed,
            'total_students': total_students
        }

    return render_template('user_profile.html', user=user, attendance_stats=attendance_stats, admin_stats=admin_stats)

@app.route('/settings', methods=['GET', 'POST'])
def user_settings():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = users_collection.find_one({"_id": session['user_id']})
    if not user:
        session.clear()
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Handle password change
        if 'current_password' in request.form and 'new_password' in request.form:
            current_password = request.form['current_password']
            new_password = request.form['new_password']

            # Verify current password
            if user.get("password") == hash_password(current_password):
                # Update password
                users_collection.update_one(
                    {"_id": session['user_id']},
                    {"$set": {"password": hash_password(new_password)}}
                )
                flash("Password updated successfully!")
            else:
                flash("Current password is incorrect!")

        # Handle profile updates
        if 'name' in request.form:
            name = request.form['name'].strip()
            if name:
                users_collection.update_one(
                    {"_id": session['user_id']},
                    {"$set": {"name": name}}
                )
                # Update session
                session['name'] = name
                flash("Profile updated successfully!")

    return render_template('user_settings.html', user=user)

@app.route('/attendance/history')
def attendance_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if session['role'] == 'student':
        # Get student's attendance records
        records = list(attendance_collection.find({"student_id": session['user_id']}).sort("date", -1))

        # Enhance records with class names
        for record in records:
            class_code = record.get('class_code')
            if class_code:
                class_data = classes_collection.find_one({"code": class_code})
                if class_data:
                    record['class_name'] = class_data.get('name', 'Unknown Class')

        return render_template('attendance_history.html', records=records)

    elif session['role'] in ['admin', 'teacher']:
        # For admins/teachers, show a different view with filtering options
        class_filter = request.args.get('class')
        date_filter = request.args.get('date')

        # Build query
        query = {}
        if class_filter:
            query['class_code'] = class_filter
        if date_filter:
            query['date'] = {"$regex": date_filter}

        # Get records
        records = list(attendance_collection.find(query).sort("date", -1).limit(100))

        # Get classes for filter dropdown
        classes = list(classes_collection.find({}))

        return render_template('attendance_history.html', records=records, classes=classes)

# SocketIO handlers
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
        if not PYZBAR_AVAILABLE:
            # If pyzbar is not available, inform the user
            logger.warning("QR code scanning is disabled because pyzbar is not available")
            emit('message', "Error: QR code scanning is disabled on this server. Please contact the administrator.")
            return

        qr_data = scan_qr(frame)
        if not qr_data:
            # Don't emit an error for every frame without a QR code
            return

        # Check if this is a class session QR code
        try:
            # Log the QR data for debugging
            logger.info(f"Processing QR data: '{qr_data}'")

            # Class session QR codes are in the format: class_code_timestamp
            if '_' in qr_data:
                session_id = qr_data.strip()
                logger.info(f"Looking up session ID: '{session_id}'")

                # Try to find the session in the database
                class_session = db["class_sessions"].find_one({"_id": session_id})

                # If not found, try again with common character replacements
                # (sometimes QR scanners can misinterpret characters)
                if not class_session:
                    logger.warning(f"Session ID not found directly: '{session_id}'. Trying alternatives...")
                    # Try with different underscore characters
                    alt_session_id = session_id.replace('_', '-')
                    class_session = db["class_sessions"].find_one({"_id": alt_session_id})

                    if not class_session:
                        # Try extracting just the class code part
                        parts = session_id.split('_', 1)
                        if len(parts) > 0:
                            class_code = parts[0]
                            # Find the most recent session for this class
                            logger.info(f"Trying to find most recent session for class: {class_code}")
                            class_sessions = list(db["class_sessions"].find(
                                {"_id": {"$regex": f"^{class_code}_"}}
                            ).sort("created_at", -1).limit(1))

                            if class_sessions:
                                class_session = class_sessions[0]
                                logger.info(f"Found alternative session: {class_session['_id']}")

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

            # Provide more specific error message
            if '_' not in qr_data:
                emit('message', "Error: Invalid QR code format. Expected class_code_timestamp format.")
            elif len(qr_data.split('_')) != 2:
                emit('message', "Error: QR code has incorrect format. Please scan a valid class QR code.")
            else:
                emit('message', "Error: QR code not found in the system. Please scan a valid class QR code.")

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
                base_distance = 20  # Base allowed distance
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

            # Send a success message with detailed information
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
            base_distance = 2  # Base allowed distance
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

# Create default admin user if none exists
if not users_collection.find_one({"role": "admin"}):
    users_collection.insert_one({
        "_id": "admin1",
        "name": "Admin User",
        "role": "admin",
        "password": hash_password("admin123")
    })
    logger.info("Created default admin user (admin1/admin123)")

if __name__ == "__main__":
    # Get host and port from environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    
    # Try to run on the specified port, if it fails, try alternative ports
    max_port_attempts = 10
    for port_attempt in range(max_port_attempts):
        try:
            if port_attempt > 0:
                port = port + port_attempt
                
            # Print a custom message showing the localhost URL
            print(f"\n* Flask application running at: http://localhost:{port}")
            print(f"* To access the application, open your browser and navigate to: http://localhost:{port}\n")
            
            # Run the app with environment-based configuration
            socketio.run(app, debug=False, host=host, port=port, log_output=False, server='eventlet')

            break
        except OSError as e:
            if "address already in use" in str(e).lower() and port_attempt < max_port_attempts - 1:
                print(f"Port {port} is already in use, trying port {port + 1}...")
                continue
            else:
                print(f"Error starting server: {e}")
                raise