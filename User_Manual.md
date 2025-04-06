# Smart Attendance System - User Manual

## Table of Contents
1. [System Overview](#system-overview)
2. [Installation & Setup](#installation--setup)
3. [User Roles](#user-roles)
4. [Admin Guide](#admin-guide)
5. [Student Guide](#student-guide)
6. [Troubleshooting](#troubleshooting)

## System Overview
The Smart Attendance System is a Flask-based web application that provides:
- QR code-based attendance tracking
- Facial recognition verification
- GPS location verification
- Real-time attendance monitoring
- Comprehensive reporting

## Installation & Setup

### Prerequisites
- Python 3.8+
- MongoDB
- Node.js (for optional frontend components)
- Webcam (for facial recognition)

### Installation Steps
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   ```
   SECRET_KEY=your_secret_key
   MONGO_URI=mongodb://localhost:27017/
   MONGO_DB_NAME=smart_attendance
   DEBUG=True
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## User Roles

### Admin
- Create/manage classes
- Register users
- Generate QR codes
- View attendance reports
- Manage system settings

### Student
- Mark attendance via QR code
- View attendance history
- View class schedule
- Update profile

## Admin Guide

### Login Process
1. Access the login page at `/login`
2. Enter your admin credentials:
   - Username (typically your email)
   - Password
3. Complete any additional verification if enabled
4. You'll be redirected to the Admin Dashboard

### User Management
1. **Register New Users**:
   - Navigate to Admin Dashboard → Register
   - Fill in user details:
     * ID (unique identifier)
     * Full Name
     * Email
     * Role (Admin/Student)
     * Department (if applicable)
   - For students: capture facial data through the web interface

2. **Manage Classes** (via manage_classes.html):
   - Create new classes with:
     * Class name and code
     * Location data (building, room, GPS coordinates)
     * Schedule (days/times)
     * Instructor assignment
   - Edit existing classes
   - Enroll/remove students
   - View class rosters

### Attendance Management
1. **Generate QR Codes**:
   - Select class and session from dropdown
   - Set validity period (default 15 minutes)
   - Choose display option:
     * Project full-screen in class
     * Download as PNG
     * Print for physical display

2. **View Reports**:
   - Filter by:
     * Date range
     * Student/Class/Section
     * Attendance status
   - Export options:
     * CSV for spreadsheets
     * PDF for printing
     * JSON for integration

### Admin Dashboard Features
The admin dashboard provides:
- Real-time statistics on:
  * Total users (admins/students)
  * Active classes
  * Today's attendance rate
- Recent system activity log
- Important notifications (expiring QR codes, etc.)
- Quick navigation to:
  * User management
  * Class management
  * Attendance reports
  * System configuration

## Student Guide

### Login Process
1. Access the login page at `/login`
2. Enter your student credentials:
   - Student ID
   - Password
3. Complete facial verification if required
4. You'll be redirected to the Student Dashboard

### Marking Attendance (via mark_attendance.html)
1. Scan class QR code using web interface:
   - Allow camera access
   - Center QR code in viewfinder
2. Complete facial verification:
   - Look directly at camera
   - Remove obstructions (glasses/hats)
   - Maintain neutral expression
3. Verify location (if enabled):
   - Allow GPS access
   - Must be within 50m of class location
4. Attendance is automatically recorded

### Viewing Information
1. **Attendance History**:
   - View all marked attendance with timestamps
   - Filter by:
     * Date range
     * Class
     * Status (present/late/absent)
   - Export personal attendance record

2. **Class Schedule**:
   - Daily/weekly view options
   - Class details:
     * Time/location
     * Instructor
     * Current attendance status
   - Set reminders for upcoming classes

### User Profile Management
Students can manage their profile through:
1. **Personal Information**:
   - Update contact details (email, phone)
   - Change profile picture
   - Reset password
2. **Security Settings**:
   - View/update facial recognition data
   - Manage two-factor authentication
3. **Preferences**:
   - Notification settings (email/push)
   - Linked devices management
   - Privacy controls

## Troubleshooting

### Common Issues
1. **QR Code Not Scanning**:
   - Ensure good lighting
   - Hold steady for 2-3 seconds
   - Check code validity period

2. **Face Verification Fails**:
   - Remove glasses/hats
   - Ensure good lighting
   - Face camera directly

3. **Location Errors**:
   - Enable GPS on device
   - Move closer to classroom
   - Check internet connection

### Support
For additional help, contact system administrator.
