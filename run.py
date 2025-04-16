import os
import sys
import socket
import time

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port=5000, max_attempts=10):
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    return None

def run_app():
    # Get port from environment or find an available one
    default_port = int(os.environ.get('PORT', 5000))

    if is_port_in_use(default_port):
        print(f"Port {default_port} is already in use. Finding an available port...")
        port = find_available_port(default_port)
        if port is None:
            print("Could not find an available port. Please close other applications and try again.")
            return
        os.environ['PORT'] = str(port)
    else:
        port = default_port

    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    # Print a nice header
    print("\n" + "=" * 70)
    print("  SMART ATTENDANCE SYSTEM - FLASK APPLICATION")
    print("=" * 70)
    print(f"\n* Application running at: http://localhost:{port}")
    print("* Open your browser and navigate to the URL above to access the system")
    print("\n* Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")

    # Run the application
    os.system(f"python app.py")

if __name__ == "__main__":
    run_app()
