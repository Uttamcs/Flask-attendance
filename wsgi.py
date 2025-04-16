import os
from app import app, socketio

if __name__ == "__main__":
    # Get host and port from environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))

    # Print a custom message showing the localhost URL
    print(f"\n* Flask application running at: http://localhost:{port}")
    print(f"* To access the application, open your browser and navigate to: http://localhost:{port}\n")

    # Run the app
    socketio.run(app, host=host, port=port)
