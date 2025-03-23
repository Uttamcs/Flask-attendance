import os
import cv2
import qrcode
import pymongo
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from hashlib import sha256
import logging
from pyzbar.pyzbar import decode
import time

# Flask app setup
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure key in production
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'qr_codes'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'face_data'), exist_ok=True)

# MongoDB setup
try:
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    logging.info("MongoDB connection successful.")
    client.server_info()
    db = client["smart_attendance"]
    users_collection = db["users"]
    attendance_collection = db["attendance"]
    classes_collection = db["classes"]
    schedules_collection = db["schedules"]
    enrollments_collection = db["enrollments"]
except pymongo.errors.ConnectionError as e:
    logging.error(f"MongoDB connection failed: {e}")
    exit(1)

# Utility functions (unchanged)
def hash_password(password):
    return sha256(password.encode()).hexdigest()

def capture_face(user_id):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Unable to access camera!")
        flash("Unable to access camera!")
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
            cv2.imshow("Register Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if face_count == 0:
        flash("No faces captured!")
        return None
    return face_paths

def scan_qr():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        flash("Unable to access camera!")
        return None
    timeout = time.time() + 30
    qr_data = None
    try:
        while True:
            if time.time() >= timeout:
                break
            ret, frame = cap.read()
            if not ret:
                continue
            decoded = decode(frame)
            if decoded:
                qr_data = decoded[0].data.decode("utf-8")
                break
            cv2.imshow("Scan QR Code", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if qr_data is None:
        flash("QR scan timed out or cancelled!")
    return qr_data

def verify_face(stored_paths):
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
        flash("No valid face data in stored images!")
        return False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        flash("Unable to access camera!")
        return False
    timeout = time.time() + 30
    face_verified = False
    try:
        while True:
            if time.time() >= timeout:
                break
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                live_face = gray[y:y+h, x:x+w]
                live_features = extract_hog_features(live_face)
                if live_features is not None:
                    similarities = [compare_features(stored_features, live_features) for stored_features in stored_features_list]
                    if np.mean(similarities) > 0.78:
                        face_verified = True
                        break
            if face_verified:
                break
            cv2.imshow("Verify Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if not face_verified:
        flash("Face verification failed or timed out!")
    return face_verified

def extract_hog_features(face):
    try:
        face_resized = cv2.resize(face, (64, 64))
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        return hog.compute(face_resized)
    except Exception as e:
        logging.error(f"Error computing HOG features: {e}")
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
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('student_dashboard'))
        flash('Invalid credentials!')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    return render_template('admin_dashboard.html', name=session['name'])

@app.route('/student/dashboard')
def student_dashboard():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    return render_template('student_dashboard.html', name=session['name'])

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
                "created_by": session['user_id']
            }
            schedules_collection.insert_one(schedule_data)
            flash(f"Class {class_data['name']} scheduled!")
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
                {"$set": {"student_id": sid, "class_code": class_code}},
                upsert=True
            )
            if result.matched_count or result.upserted_id:
                enrolled_count += 1
        
        flash(f"Enrolled {enrolled_count} students from section {section} in class {class_code}!")
        return redirect(url_for('admin_dashboard'))
    
    return render_template('enroll_students.html', classes=classes, sections=sections)

@app.route('/student/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    if request.method == 'POST':
        qr_data = scan_qr()
        if not qr_data:
            return redirect(url_for('mark_attendance'))
        student_id = qr_data.strip()
        student = users_collection.find_one({"_id": student_id, "role": "student"})
        if not student or student_id != session['user_id']:
            flash("Invalid student QR code!")
            return redirect(url_for('mark_attendance'))
        
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
            flash("No active class found at this time!")
            return redirect(url_for('mark_attendance'))

        # Check if attendance is already marked for this student in this class session
        existing_attendance = attendance_collection.find_one({
            "student_id": student_id,
            "class_code": active_schedule["class_code"],
            "date": {
                "$gte": active_schedule["start_time"],
                "$lte": active_schedule["end_time"]
            }
        })
        if existing_attendance:
            flash("Attendance already marked for this class session!")
            return redirect(url_for('mark_attendance'))

        if not verify_face(student["face_data"]):
            return redirect(url_for('mark_attendance'))
        if not verify_gps():
            flash("GPS verification failed! You must be in the classroom.")
            return redirect(url_for('mark_attendance'))

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
        flash(f"Attendance marked for {student['name']} at {date}!")
        return redirect(url_for('student_dashboard'))
    return render_template('mark_attendance.html')

@app.route('/student/view_attendance')
def view_student_attendance():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    records = list(attendance_collection.find({"student_id": session['user_id']}))
    return render_template('view_student_attendance.html', attendance=records)

@app.route('/student/view_schedule')
def view_class_schedule():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    enrollments = enrollments_collection.find({"student_id": session['user_id']})
    schedule = []
    for enrollment in enrollments:
        scheds = schedules_collection.find({"class_code": enrollment["class_code"]})
        for sched in scheds:
            sched["class_name"] = sched.get("class_name", classes_collection.find_one({"code": enrollment["class_code"]})["name"])
            schedule.append(sched)
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
            records = list(attendance_collection.find(query))
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
    return render_template('view_attendance_reports.html')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not users_collection.find_one({"role": "admin"}):
        users_collection.insert_one({
            "_id": "admin1",
            "name": "Admin User",
            "role": "admin",
            "password": hash_password("admin123")
        })
    app.run(debug=True)