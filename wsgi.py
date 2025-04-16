import os
import logging
from app import app, socketio

# Disable all logging for cleaner output
logging.getLogger('werkzeug').disabled = True
logging.getLogger('engineio').disabled = True
logging.getLogger('socketio').disabled = True

# This is used by gunicorn
app = socketio.run_wsgi(app)

if __name__ == "__main__":
    # Get host and port from environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))

    # Print a custom message showing the localhost URL
    print(f"\n* Flask application running at: http://localhost:{port}")
    print(f"* To access the application, open your browser and navigate to: http://localhost:{port}\n")

    # Run the app
    socketio.run(app, host=host, port=port, log_output=False)
