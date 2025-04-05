# Smart Attendance System

A facial recognition and QR code-based attendance system built with Flask, OpenCV, and MongoDB.

## Features

- Face recognition for student verification
- QR code generation for class sessions
- Location-based verification
- Admin dashboard for managing classes and students
- Student dashboard for viewing attendance records
- Real-time attendance marking with Socket.IO

## Prerequisites

- Python 3.10+
- MongoDB
- OpenCV dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-attendance-system.git
cd smart-attendance-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

5. Edit the `.env` file with your configuration.

## Running Locally

```bash
python app.py
```

The application will be available at http://localhost:5000

## Deployment Options

### Heroku

1. Create a Heroku account and install the Heroku CLI
2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Add MongoDB add-on:
```bash
heroku addons:create mongolab:sandbox
```

5. Set environment variables:
```bash
heroku config:set SECRET_KEY=your-secret-key
heroku config:set DEBUG=False
```

6. Deploy to Heroku:
```bash
git push heroku main
```

### Railway

1. Create a Railway account
2. Create a new project and select "Deploy from GitHub"
3. Connect your GitHub repository
4. Add a MongoDB plugin
5. Set environment variables in the Railway dashboard
6. Deploy your application

### PythonAnywhere

1. Create a PythonAnywhere account
2. Upload your code or clone from GitHub
3. Set up a virtual environment and install dependencies
4. Configure a web app with Flask
5. Set up environment variables
6. Start your application

## Default Admin User

- Username: admin1
- Password: admin123

**Important:** Change the default admin password after first login!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
