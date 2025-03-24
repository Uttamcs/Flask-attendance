import os
import cv2
import qrcode
import pymongo
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
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
        decoded = decode(frame)
        if decoded:
            logger.info(f"QR code detected: {decoded[0].data.decode('utf-8')}")
            return decoded[0].data.decode("utf-8")
        return None
    except Exception as e:
        logger.error(f"Error decoding QR: {e}")
        return None

def verify_face(stored_paths, live_frame):
    try:
        stored_features_list = []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        for stored_path in stored_paths:
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
                stored_features_list.append(features)
        if not stored_features_list:
            logger.warning("No valid face data in stored images")
            return False

        gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            logger.debug("No face detected in live frame")
            return False
        for (x, y, w, h) in faces:
            live_face = gray[y:y+h, x:x+w]
            live_features = extract_hog_features(live_face)
            if live_features is not None:
                similarities = [compare_features(stored_features, live_features) for stored_features in stored_features_list]
                avg_similarity = np.mean(similarities)
                logger.debug(f"Face similarity: {avg_similarity}")
                if avg_similarity > 0.78:
                    logger.info("Face verified successfully")
                    return True
        return False
    except Exception as e:
        logger.error(f"Error in face verification: {e}")
        return False

def extract_hog_features(face):
    try:
        face_resized = cv2.resize(face, (64, 64))
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        return hog.compute(face_resized)
    except Exception as e:
        logger.error(f"Error computing HOG features: {e}")
        return None

def compare_features(features1, features2):
    if features1 is None or features2 is None or features1.shape != features2.shape:
        return 0.0
    features1 = features1 / (np.linalg.norm(features1) + 1e-6)
    features2 = features2 / (np.linalg.norm(features2) + 1e-6)
    return np.dot(features1, features2)

def verify_gps():
    classroom_lat, classroom_lon = 37.7749, -122.4194
    student_lat, student_lon = 37.7750, -122.4195
    R = 6371e3
    φ1, φ2 = np.radians(classroom_lat), np.radians(student_lat)
    Δφ = np.radians(student_lat - classroom_lat)
    Δλ = np.radians(student_lon - classroom_lon)
    a = np.sin(Δφ/2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ/2) ** 2
    c = 2 * np.atan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    logger.debug(f"GPS distance: {distance} meters")
    return distance < 50

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
            face_paths = capture_face(data["_id"])
            if not face_paths:
                return redirect(url_for('register'))
            data["face_data"] = face_paths
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(data["_id"])
            qr.make(fit=True)
            qr_img = qr.make_image(fill='black', back_color='white')
            qr_path = os.path.join(app.config['UPLOAD_FOLDER'], 'qr_codes', f"qr_{data['_id']}.png")
            qr_img.save(qr_path)
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
    logger.debug("Received video frame, size: %d bytes", len(data))
    if 'user_id' not in session or session['role'] != 'student':
        logger.warning("Unauthorized WebSocket access")
        emit('message', "Error: Unauthorized access")
        return

    try:
        img_data = base64.b64decode(data.split(',')[1])
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        logger.debug("Frame decoded successfully")
    except Exception as e:
        logger.error("Error decoding frame: %s", e)
        emit('message', "Error: Failed to process video frame")
        return

    qr_data = scan_qr(frame)
    if not qr_data:
        logger.debug("No QR code detected in frame")
        return
    student_id = qr_data.strip()
    logger.info("QR code scanned: %s", student_id)
    student = users_collection.find_one({"_id": student_id, "role": "student"})
    if not student or student_id != session['user_id']:
        logger.warning("Invalid QR code: %s", student_id)
        emit('message', "Error: Invalid student QR code")
        return

    now = datetime.now()
    schedules = schedules_collection.find({
        "start_time": {"$lte": now.strftime("%Y-%m-%d %H:%M:%S")},
        "end_time": {"$gte": now.strftime("%Y-%m-%d %H:%M:%S")}
    })
    active_schedule = None
    for sched in schedules:
        if enrollments_collection.find_one({"student_id": student_id, "class_code": sched["class_code"]}):
            active_schedule = sched
            break
    if not active_schedule:
        logger.warning("No active class schedule found")
        emit('message', "Error: No active class found at this time")
        return

    existing_attendance = attendance_collection.find_one({
        "student_id": student_id,
        "class_code": active_schedule["class_code"],
        "date": {
            "$gte": active_schedule["start_time"],
            "$lte": active_schedule["end_time"]
        }
    })
    if existing_attendance:
        logger.info("Attendance already marked for %s in %s", student_id, active_schedule['class_code'])
        emit('message', "Error: Attendance already marked for this class session")
        return

    if not verify_face(student["face_data"], frame):
        logger.warning("Face verification failed for %s", student_id)
        emit('message', "Error: Face verification failed")
        return

    if not verify_gps():
        logger.warning("GPS verification failed for %s", student_id)
        emit('message', "Error: GPS verification failed! You must be in the classroom")
        return

    date = now.strftime("%Y-%m-%d %H:%M:%S")
    class_name = active_schedule.get("class_name", "Unknown Class")
    attendance_data = {
        "student_id": student_id,
        "class_code": active_schedule["class_code"],
        "class_name": class_name,
        "date": date,
        "status": "Present",
        "gps_verified": True,
        "schedule_start_time": active_schedule["start_time"],
        "schedule_end_time": active_schedule["end_time"]
    }
    attendance_collection.insert_one(attendance_data)
    logger.info("Attendance marked for %s in %s at %s", student_id, class_name, date)
    emit('message', "Attendance Marked")

if __name__ == "__main__":
    if not users_collection.find_one({"role": "admin"}):
        users_collection.insert_one({
            "_id": "admin1",
            "name": "Admin User",
            "role": "admin",
            "password": hash_password("admin123")
        })
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)