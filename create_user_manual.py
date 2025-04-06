import os
from fpdf import FPDF
from datetime import datetime

class PDF(FPDF):
    def header(self):
        # Logo
        self.image('static/college_logo.png', 10, 8, 20)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(20)
        # Title
        self.cell(170, 10, 'Smart Attendance System - User Manual', 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        # Date
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d")}', 0, 0, 'R')

    def chapter_title(self, title):
        # Arial 12
        self.set_font('Arial', 'B', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, title, 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self, body):
        # Times 12
        self.set_font('Arial', '', 10)
        # Output justified text
        self.multi_cell(0, 5, body)
        # Line break
        self.ln()

    def add_section(self, title, content):
        self.chapter_title(title)
        self.chapter_body(content)

    def add_subsection(self, title):
        # Arial 12
        self.set_font('Arial', 'B', 11)
        # Title
        self.cell(0, 6, title, 0, 1, 'L')
        # Line break
        self.ln(4)

    def add_bullet_point(self, text):
        self.set_font('Arial', '', 10)
        self.cell(5, 5, chr(149), 0, 0, 'C')  # bullet character
        self.cell(0, 5, text, 0, 1)

def create_user_manual():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Introduction
    pdf.add_section("1. Introduction", 
        "The Smart Attendance System is a modern, secure solution for tracking student attendance in educational institutions. "
        "It uses advanced technologies like facial recognition and GPS location verification to ensure accurate attendance records. "
        "This manual provides comprehensive guidance on how to use the system for both administrators and students.")
    
    # System Requirements
    pdf.add_section("2. System Requirements", 
        "To use the Smart Attendance System, you need:\n\n"
        "- A computer or mobile device with a modern web browser (Chrome, Firefox, Safari, or Edge)\n"
        "- A working camera for facial recognition\n"
        "- Location services enabled for GPS verification\n"
        "- Internet connection\n"
        "- For administrators: Access to the admin credentials")
    
    # Getting Started
    pdf.add_section("3. Getting Started", "")
    
    # Login
    pdf.add_subsection("3.1 Logging In")
    pdf.chapter_body(
        "1. Navigate to the login page\n"
        "2. Enter your User ID and Password\n"
        "3. Click the Login button\n"
        "4. The system will redirect you to your dashboard based on your role (admin or student)")
    
    # Admin Dashboard
    pdf.add_page()
    pdf.add_section("4. Administrator Guide", "")
    
    pdf.add_subsection("4.1 Admin Dashboard")
    pdf.chapter_body(
        "The admin dashboard provides an overview of the system and quick access to all administrative functions. "
        "From here, you can manage users, classes, schedules, and view attendance reports.")
    
    # User Registration
    pdf.add_subsection("4.2 Registering New Users")
    pdf.chapter_body(
        "To register a new user (student or admin):\n\n"
        "1. Click on 'Register User' from the admin dashboard\n"
        "2. Fill in the required information (User ID, Name, Password, Role)\n"
        "3. For student registration, you'll be prompted to capture face data\n"
        "4. The system will open the camera directly on the webpage\n"
        "5. A countdown will show the remaining time/photos to capture (50 photos required)\n"
        "6. Ensure the student's face is clearly visible and well-lit\n"
        "7. Once complete, the student account will be created with facial recognition data")
    
    # Class Management
    pdf.add_subsection("4.3 Managing Classes")
    pdf.chapter_body(
        "To create and manage classes:\n\n"
        "1. Click on 'Create Class' from the admin dashboard\n"
        "2. Enter the Class Name and a unique Class Code\n"
        "3. The system will capture your current location as the classroom location\n"
        "4. Click Submit to create the class\n\n"
        "To manage existing classes:\n\n"
        "1. Click on 'Manage Classes' from the admin dashboard\n"
        "2. View all classes in the system\n"
        "3. Delete classes as needed (this will also remove all related schedules and attendance records)")
    
    # Scheduling Classes
    pdf.add_subsection("4.4 Scheduling Classes")
    pdf.chapter_body(
        "To schedule a class session:\n\n"
        "1. Click on 'Schedule Class' from the admin dashboard\n"
        "2. Select the Class from the dropdown menu\n"
        "3. Enter the Start Time and End Time\n"
        "4. Click Submit to create the schedule")
    
    # Enrolling Students
    pdf.add_subsection("4.5 Enrolling Students")
    pdf.chapter_body(
        "To enroll students in classes:\n\n"
        "1. Click on 'Enroll Students' from the admin dashboard\n"
        "2. Select the Class from the dropdown menu\n"
        "3. Select the Student from the dropdown menu\n"
        "4. Click Submit to enroll the student")
    
    # Generating QR Codes
    pdf.add_page()
    pdf.add_subsection("4.6 Generating QR Codes for Attendance")
    pdf.chapter_body(
        "To generate a QR code for a class session:\n\n"
        "1. Click on 'Generate QR Code' from the admin dashboard\n"
        "2. Select the Class from the dropdown menu\n"
        "3. Enter a Session Name (e.g., 'Lecture 1', 'Lab 2')\n"
        "4. Set the validity period in minutes\n"
        "5. Click Generate\n"
        "6. The system will display the QR code which can be downloaded or shared with students\n"
        "7. Students must scan this QR code to mark their attendance within the validity period")
    
    # Viewing Attendance Reports
    pdf.add_subsection("4.7 Viewing Attendance Reports")
    pdf.chapter_body(
        "To view attendance reports:\n\n"
        "1. Click on 'View Reports' from the admin dashboard\n"
        "2. Use the filters to select specific classes or dates\n"
        "3. View the attendance records for all students\n"
        "4. The report shows attendance status, verification methods used, and timestamps")
    
    # Student Guide
    pdf.add_page()
    pdf.add_section("5. Student Guide", "")
    
    # Student Dashboard
    pdf.add_subsection("5.1 Student Dashboard")
    pdf.chapter_body(
        "The student dashboard provides an overview of your upcoming classes and quick access to attendance functions. "
        "From here, you can mark attendance, view your attendance history, and check your class schedule.")
    
    # Marking Attendance
    pdf.add_subsection("5.2 Marking Attendance")
    pdf.chapter_body(
        "To mark your attendance for a class:\n\n"
        "1. Click on 'Mark Attendance' from the student dashboard\n"
        "2. Allow camera access when prompted\n"
        "3. The system will display a progress bar showing the verification steps\n"
        "4. Scan the QR code displayed by your instructor\n"
        "5. Once the QR code is detected, the system will proceed to face verification\n"
        "6. Look directly at the camera and ensure your face is clearly visible\n"
        "7. The system will display a similarity score during face verification\n"
        "8. After successful face verification, the system will verify your location\n"
        "9. You must be within 20 meters of the classroom location to pass verification\n"
        "10. Once all verifications are complete, your attendance will be marked automatically\n"
        "11. The system will display a confirmation message")
    
    # Viewing Attendance History
    pdf.add_subsection("5.3 Viewing Attendance History")
    pdf.chapter_body(
        "To view your attendance history:\n\n"
        "1. Click on 'View Attendance' from the student dashboard\n"
        "2. The system will display all your attendance records\n"
        "3. You can see the class name, date, attendance status, and verification methods used")
    
    # Viewing Class Schedule
    pdf.add_subsection("5.4 Viewing Class Schedule")
    pdf.chapter_body(
        "To view your class schedule:\n\n"
        "1. Click on 'View Schedule' from the student dashboard\n"
        "2. The system will display all your scheduled classes\n"
        "3. You can see the class name, start time, and end time")
    
    # Troubleshooting
    pdf.add_page()
    pdf.add_section("6. Troubleshooting", "")
    
    # Face Verification Issues
    pdf.add_subsection("6.1 Face Verification Issues")
    pdf.chapter_body(
        "If you're having trouble with face verification:\n\n"
        "1. Ensure you are in a well-lit environment\n"
        "2. Remove any face coverings (masks, sunglasses, etc.)\n"
        "3. Look directly at the camera\n"
        "4. Keep your face centered in the frame\n"
        "5. If problems persist, contact your administrator to update your face data")
    
    # QR Code Scanning Issues
    pdf.add_subsection("6.2 QR Code Scanning Issues")
    pdf.chapter_body(
        "If you're having trouble scanning QR codes:\n\n"
        "1. Ensure the QR code is clearly visible and not damaged\n"
        "2. Hold your device steady and ensure the QR code is in frame\n"
        "3. Make sure there is adequate lighting\n"
        "4. Try moving closer to the QR code\n"
        "5. If the QR code is displayed on a screen, adjust the screen brightness")
    
    # Location Verification Issues
    pdf.add_subsection("6.3 Location Verification Issues")
    pdf.chapter_body(
        "If you're having trouble with location verification:\n\n"
        "1. Ensure location services are enabled on your device\n"
        "2. Allow the browser to access your location when prompted\n"
        "3. Make sure you are physically present in the classroom\n"
        "4. If you're in a building with poor GPS signal, try moving closer to a window\n"
        "5. If problems persist, contact your administrator")
    
    # Login Issues
    pdf.add_subsection("6.4 Login Issues")
    pdf.chapter_body(
        "If you're having trouble logging in:\n\n"
        "1. Ensure you're using the correct User ID and Password\n"
        "2. Check if Caps Lock is enabled\n"
        "3. Clear your browser cache and cookies\n"
        "4. Try using a different browser\n"
        "5. If problems persist, contact your administrator to reset your password")
    
    # Security Features
    pdf.add_page()
    pdf.add_section("7. Security Features", 
        "The Smart Attendance System incorporates multiple security features to ensure accurate attendance records:\n\n"
        "1. Facial Recognition: Verifies the identity of students using their unique facial features\n"
        "2. GPS Location Verification: Ensures students are physically present in the classroom\n"
        "3. QR Code Authentication: Provides a unique, time-limited token for each class session\n"
        "4. Password Protection: Secures user accounts with hashed passwords\n"
        "5. Role-Based Access Control: Restricts access to features based on user roles")
    
    # Best Practices
    pdf.add_section("8. Best Practices", 
        "For optimal use of the Smart Attendance System:\n\n"
        "1. Administrators should generate QR codes shortly before class begins\n"
        "2. Set appropriate validity periods for QR codes (typically the duration of the class)\n"
        "3. Students should arrive a few minutes early to mark attendance\n"
        "4. Ensure good lighting conditions for facial recognition\n"
        "5. Keep your face data up-to-date (contact administrator if your appearance changes significantly)\n"
        "6. Regularly check your attendance records for accuracy")
    
    # Save the PDF
    pdf.output('Smart_Attendance_System_User_Manual.pdf')
    print("User manual created successfully!")

if __name__ == "__main__":
    create_user_manual()
