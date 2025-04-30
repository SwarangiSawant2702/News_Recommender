import subprocess
import time
import webbrowser

# Step 1: Run main.py to fetch and clean news
print("🚀 Running main.py to fetch and clean news...")
subprocess.run(["python", "main.py"])

# Step 2: Start Flask app (non-blocking)
print("🌐 Launching Flask app...")
flask_process = subprocess.Popen(["python", "flaskapp.py"])

# Step 3: Wait for Flask to start and open the browser
print("⏳ Waiting for Flask to start...")
time.sleep(20)  # Wait for Flask to initialize, adjust time if needed
webbrowser.open("http://127.0.0.1:5000")  # Open the default browser

# Optionally, keep script alive (or handle Flask's lifecycle)
flask_process.wait()  # Wait for Flask to exit if needed
print("✅ Scheduler script complete.")
