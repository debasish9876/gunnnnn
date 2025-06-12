from flask import Flask, request, jsonify, render_template_string, Response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
from datetime import datetime
from twilio.rest import Client
import requests
import json

app = Flask(__name__)
CORS(app)

class CriminalDetectionSystem:
    def __init__(self):
        # Load YOUR trained model - exactly like your main.py
        try:
            self.model = YOLO(r'C:\Users\USER\Desktop\wepon1\Weapon-Detection-Using-Yolov8\Weapon detection\best.pt')
            print("✅ Your trained model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading your model: {e}")
            self.model = None
        
        # SMS Configuration - REPLACE WITH YOUR ACTUAL CREDENTIALS
        # ⚠️ IMPORTANT: Verify these credentials in your Twilio Console
        self.twilio_sid = 'ACe9fbc5b8e477ebae9414f2bd40fc29b9'  # Account SID
        self.twilio_token = '58a12c0928d0971b7eca147659c49499'  # Auth Token
        self.twilio_phone = '+15854886903'  # Your Twilio phone number
        
        # ⚠️ IMPORTANT: These numbers must be verified in Twilio Console (for trial accounts)
        # Replace with YOUR verified phone numbers:
        self.police_numbers = ['+917848084152', '+919117597217']  # Recipient numbers
        
        # 🔧 DEBUGGING: Uncomment to test with a single verified number first
        # self.police_numbers = ['+91XXXXXXXXXX']  # Replace with your verified number
        
        # Print configuration for debugging
        print(f"📞 Twilio SID: {self.twilio_sid[:10]}...")
        print(f"📞 From Number: {self.twilio_phone}")
        print(f"📞 To Numbers: {self.police_numbers}")
        print("⚠️  Make sure recipient numbers are verified in Twilio Console!")
        
        # Detection settings - EXACTLY like your main.py
        self.confidence_threshold = 0.6
        self.img_size = 640
        
        # System state
        self.detection_active = False
        self.total_detections = 0
        self.alerts_sent = 0
        self.last_alert_time = 0
        self.alert_cooldown = 5  # Reduced from 10 to 5 seconds
        self.current_location = None
        
        # Add detection tracking
        self.last_detection_time = 0
        self.detection_count_session = 0
        
        # Create directories for storing footage
        self.footage_dir = "detected_footage"
        self.create_directories()
        
        # Get current location on startup
        self.get_current_location()
        
    def create_directories(self):
        """Create necessary directories for storing footage"""
        if not os.path.exists(self.footage_dir):
            os.makedirs(self.footage_dir)
            print(f"📁 Created footage directory: {self.footage_dir}")
    
    def get_current_location(self):
        """Get current location using IP geolocation"""
        try:
            # Using a free IP geolocation service
            response = requests.get('http://ip-api.com/json/', timeout=10)
            data = response.json()
            
            if data['status'] == 'success':
                self.current_location = {
                    'latitude': data['lat'],
                    'longitude': data['lon'],
                    'city': data['city'],
                    'region': data['regionName'],
                    'country': data['country']
                }
                print(f"📍 Location detected: {data['city']}, {data['regionName']}")
                print(f"🌍 Coordinates: {data['lat']}, {data['lon']}")
            else:
                # Fallback coordinates (you can set your specific location here)
                self.current_location = {
                    'latitude': 22.2587,  # Rourkela coordinates
                    'longitude': 84.8854,
                    'city': 'Rourkela',
                    'region': 'Odisha',
                    'country': 'India'
                }
                print("📍 Using fallback location: Rourkela, Odisha")
                
        except Exception as e:
            print(f"❌ Location error: {e}")
            # Fallback coordinates
            self.current_location = {
                'latitude': 22.2587,
                'longitude': 84.8854,
                'city': 'Rourkela',
                'region': 'Odisha',
                'country': 'India'
            }
    
    def save_detection_footage(self, frame, weapon_type, confidence):
        """Save the frame when weapon is detected"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.footage_dir}/detection_{weapon_type}_{timestamp}_{confidence:.2f}.jpg"
            
            # Add detection info overlay on the frame
            overlay_frame = frame.copy()
            text = f"WEAPON DETECTED: {weapon_type} ({confidence:.2f})"
            cv2.putText(overlay_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add timestamp
            time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(overlay_frame, time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add location if available
            if self.current_location:
                location_text = f"Location: {self.current_location['latitude']:.4f}, {self.current_location['longitude']:.4f}"
                cv2.putText(overlay_frame, location_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save the frame
            success = cv2.imwrite(filename, overlay_frame)
            if success:
                print(f"💾 Footage saved: {filename}")
                return filename
            else:
                print(f"❌ Failed to save footage: {filename}")
                return None
                
        except Exception as e:
            print(f"❌ Error saving footage: {e}")
            return None
    
    def detect_objects(self, frame):
        """Detect objects using YOUR trained model - simplified approach"""
        if self.model is None:
            return []
            
        try:
            # Run prediction exactly like your main.py but without save/show
            results = self.model.predict(source=frame, imgsz=self.img_size, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.model.names[cls_id]
                        
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detection = {
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        }
                        detections.append(detection)
                        
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def send_alert(self, weapon_type, confidence=0.0, footage_path=None):
        """Send SMS alert with location and footage info - WITH DETAILED DEBUGGING"""
        print(f"🔔 Attempting to send SMS alert for {weapon_type}")
        print(f"📞 Twilio SID: {self.twilio_sid[:10]}...")
        print(f"📞 From Number: {self.twilio_phone}")
        print(f"📞 To Numbers: {self.police_numbers}")
        
        try:
            # Validate Twilio credentials first
            if not self.twilio_sid or not self.twilio_token:
                print("❌ Missing Twilio credentials!")
                return False
            
            if not self.twilio_phone:
                print("❌ Missing Twilio phone number!")
                return False
            
            if not self.police_numbers:
                print("❌ No recipient phone numbers configured!")
                return False
            
            client = Client(self.twilio_sid, self.twilio_token)
            print("✅ Twilio client created successfully")
            
            # Create shorter message to avoid SMS length limits
            location_info = ""
            if self.current_location:
                location_info = f"\nLocation: {self.current_location['city']}\nCoords: {self.current_location['latitude']:.4f}, {self.current_location['longitude']:.4f}"
            
            footage_info = ""
            if footage_path:
                footage_info = f"\nFootage: {os.path.basename(footage_path)}"
            
            # Shorter message format
            message = f"🚨 ALERT: {weapon_type} DETECTED!\nConfidence: {confidence:.1f}%\nTime: {datetime.now().strftime('%H:%M:%S')}{location_info}{footage_info}\nREQUIRES IMMEDIATE RESPONSE!"
            
            print(f"📝 Message length: {len(message)} characters")
            print(f"📝 Message content: {message}")
            
            success_count = 0
            for phone in self.police_numbers:
                try:
                    print(f"📤 Sending SMS to {phone}...")
                    sms = client.messages.create(
                        body=message,
                        from_=self.twilio_phone,
                        to=phone
                    )
                    print(f"✅ SMS SUCCESS to {phone}!")
                    print(f"📨 Message SID: {sms.sid}")
                    print(f"📊 Message Status: {sms.status}")
                    success_count += 1
                    
                except Exception as sms_error:
                    print(f"❌ SMS FAILED to {phone}: {str(sms_error)}")
                    print(f"❌ Error type: {type(sms_error).__name__}")
                    
                    # Check for common Twilio errors
                    error_str = str(sms_error).lower()
                    if "unverified" in error_str:
                        print("⚠️  PHONE NUMBER NOT VERIFIED! Verify in Twilio Console")
                    elif "insufficient funds" in error_str or "balance" in error_str:
                        print("⚠️  INSUFFICIENT TWILIO BALANCE!")
                    elif "invalid" in error_str and "from" in error_str:
                        print("⚠️  INVALID FROM NUMBER! Check Twilio phone number")
                    elif "invalid" in error_str and ("to" in error_str or "number" in error_str):
                        print(f"⚠️  INVALID TO NUMBER: {phone}")
            
            print(f"📊 SMS Results: {success_count}/{len(self.police_numbers)} sent successfully")
            
            if success_count > 0:
                self.alerts_sent += 1
                self.last_alert_time = time.time()
                print(f"✅ Alert counter updated: {self.alerts_sent}")
                return True
            else:
                print("❌ No SMS messages sent successfully")
                return False
            
        except Exception as e:
            print(f"❌ MAJOR SMS ERROR: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            
            # Check for authentication errors
            error_str = str(e).lower()
            if "authenticate" in error_str or "invalid" in error_str:
                print("⚠️  AUTHENTICATION FAILED! Check your Twilio SID and Token")
            elif "account" in error_str:
                print("⚠️  ACCOUNT ISSUE! Check your Twilio account status")
            
            return False
    
    def should_alert(self):
        """Check if enough time has passed since last alert - FIXED VERSION"""
        current_time = time.time()
        time_since_last = current_time - self.last_alert_time
        
        print(f"🕐 Alert cooldown check: {time_since_last:.1f}s since last alert (cooldown: {self.alert_cooldown}s)")
        
        if time_since_last > self.alert_cooldown:
            print("✅ Alert cooldown passed - SMS allowed")
            return True
        else:
            print(f"⏳ Alert cooldown active - {self.alert_cooldown - time_since_last:.1f}s remaining")
            return False

# Initialize system
detector = CriminalDetectionSystem()

# Global variables for video stream
camera = None
detection_thread = None
current_detections = []
frame_lock = threading.Lock()
latest_frame = None

def detection_loop():
    """Main detection loop with improved camera handling - FIXED VERSION"""
    global camera, current_detections, latest_frame
    
    # Try different camera indices
    camera_indices = [0, 1, 2]  # Try multiple camera indices
    camera = None
    
    for idx in camera_indices:
        try:
            test_camera = cv2.VideoCapture(idx)
            if test_camera.isOpened():
                ret, frame = test_camera.read()
                if ret:
                    camera = test_camera
                    print(f"🎥 Camera {idx} opened successfully")
                    break
                else:
                    test_camera.release()
            else:
                test_camera.release()
        except Exception as e:
            print(f"❌ Camera {idx} failed: {e}")
            continue
    
    if camera is None:
        print("❌ No camera found! Please check your camera connection.")
        detector.detection_active = False
        return
    
    # Configure camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for real-time
    
    print("🎥 Camera started - Detection loop running")
    
    frame_count = 0
    while detector.detection_active:
        ret, frame = camera.read()
        if not ret:
            print("❌ Failed to read frame from camera")
            time.sleep(0.1)
            continue
        
        # Update latest frame for streaming
        with frame_lock:
            latest_frame = frame.copy()
        
        # Run detection every few frames to improve performance
        frame_count += 1
        if frame_count % 2 == 0:  # Process every 2nd frame (increased frequency)
            detections = detector.detect_objects(frame)
            current_detections = detections
            
            # FIXED: If something detected, always increment detection counter first
            if detections:
                detector.total_detections += 1  # Count detection immediately
                detector.last_detection_time = time.time()
                
                print(f"🎯 DETECTION #{detector.total_detections}")
                
                # Check if we should send alert (cooldown logic)
                if detector.should_alert():
                    for det in detections:
                        weapon_name = det['class_name']
                        confidence = det['confidence'] * 100  # Convert to percentage
                        
                        print(f"🚨 WEAPON DETECTED: {weapon_name} ({confidence:.1f}%)")
                        print(f"🚨 SENDING SMS ALERT...")
                        
                        # Save footage
                        footage_path = detector.save_detection_footage(frame, weapon_name, confidence)
                        
                        # Send SMS alert - SIMPLIFIED LOGIC
                        try:
                            sms_success = detector.send_alert(weapon_name, confidence, footage_path)
                            if sms_success:
                                print(f"📱 ✅ SMS Alert sent successfully for {weapon_name}")
                            else:
                                print(f"📱 ❌ SMS Alert failed for {weapon_name}")
                        except Exception as sms_error:
                            print(f"📱 ❌ SMS Exception: {sms_error}")
                        
                        break  # Only alert once per detection cycle
                else:
                    print(f"⏳ Detection #{detector.total_detections} - SMS skipped due to cooldown")
        
        time.sleep(0.033)  # ~30 FPS
    
    if camera:
        camera.release()
        print("🎥 Camera released")

def generate_frames():
    """Generate video frames with detection boxes - FIXED VERSION"""
    global latest_frame, current_detections
    
    while detector.detection_active:
        if latest_frame is None:
            # Create a placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera Loading...", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
            continue
        
        # Get current frame safely
        with frame_lock:
            frame = latest_frame.copy()
        
        # Draw detection boxes
        for det in current_detections:
            bbox = det['bbox']
            name = det['class_name']
            conf = det['confidence']
            
            # Draw red rectangle
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            
            # Draw label with background
            label = f"{name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[0]+label_size[0], bbox[1]), (0, 0, 255), -1)
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add system info overlay
        cv2.putText(frame, f"Detections: {detector.total_detections}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"SMS Alerts: {detector.alerts_sent}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show cooldown status
        if detector.last_alert_time > 0:
            cooldown_remaining = max(0, detector.alert_cooldown - (time.time() - detector.last_alert_time))
            if cooldown_remaining > 0:
                cv2.putText(frame, f"SMS Cooldown: {cooldown_remaining:.1f}s", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Criminal Detection System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: 
#1a1a1a; color: white; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: 
#ff4444; font-size: 2.5rem; margin: 0; }
        .main { display: grid; grid-template-columns: 2fr 1fr; gap: 30px; }
        .video-section { background: #333; padding: 20px; border-radius: 10px; }
        .control-section { background: #333; padding: 20px; border-radius: 10px; }
        .video-container { width: 100%; height: 400px; background: black; border-radius: 10px; overflow: hidden; position: relative; }
        .video-stream { width: 100%; height: 100%; object-fit: cover; }
        .controls { margin-top: 20px; text-align: center; }
        .btn { padding: 15px 30px; margin: 10px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; font-weight: bold; }
        .btn-start { background: 
#4CAF50; color: white; }
        .btn-stop { background: 
#f44336; color: white; }
        .btn-test { background: 
#ff9800; color: white; }
        .btn-location { background: 
#2196F3; color: white; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .status { padding: 20px; margin: 20px 0; border-radius: 10px; }
        .status-active { background: 
#4CAF50; }
        .status-inactive { background: #666; }
        .status-alert { background: 
#ff4444; animation: blink 1s infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0.7; } }
        .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
        .stat { background: #444; padding: 15px; border-radius: 10px; text-align: center; }
        .stat-number { font-size: 2rem; font-weight: bold; color: 
#4CAF50; }
        .location-info { background: #444; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
        .location-info h3 { margin-top: 0; color: 
#2196F3; }
        .log { background: #222; padding: 15px; border-radius: 10px; height: 200px; overflow-y: auto; }
        .log-item { padding: 10px; border-bottom: 1px solid #444; }
        .log-time { color: #888; font-size: 0.9rem; }
        .log-message { color: 
#ff4444; font-weight: bold; }
        .loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #666; }
        .debug-info { background: #444; padding: 10px; border-radius: 5px; margin-top: 20px; font-size: 0.9rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 CRIMINAL DETECTION SYSTEM - FIXED</h1>
            <p>AI-Powered Weapon Detection & Alert System with Location Tracking</p>
        </div>
        
        <div class="main">
            <div class="video-section">
                <h2>📹 Live Camera Feed</h2>
                <div class="video-container">
                    <img id="videoStream" class="video-stream" src="/video_feed" style="display: none;">
                    <div id="noVideo" class="loading">Camera Inactive - Click Start Detection</div>
                </div>
                <div class="controls">
                    <button id="startBtn" class="btn btn-start" onclick="startDetection()">🎯 START DETECTION</button>
                    <button id="stopBtn" class="btn btn-stop" onclick="stopDetection()" disabled>⏹️ STOP DETECTION</button>
                    <button class="btn btn-test" onclick="testSMS()">📱 TEST SMS</button>
                    <button class="btn btn-location" onclick="updateLocation()">📍 UPDATE LOCATION</button>
                </div>
            </div>
            
            <div class="control-section">
                <div id="status" class="status status-inactive">
                    <h3>System Status: <span id="statusText">OFFLINE</span></h3>
                    <p id="statusDetails">Click Start Detection to begin monitoring</p>
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <div id="detectionCount" class="stat-number">0</div>
                        <div>Total Detections</div>
                    </div>
                    <div class="stat">
                        <div id="alertCount" class="stat-number">0</div>
                        <div>SMS Alerts Sent</div>
                    </div>
                </div>
                
                <div class="location-info">
                    <h3>📍 Current Location</h3>
                    <p id="locationText">Loading location...</p>
                    <p id="coordinatesText">Coordinates: Loading...</p>
                </div>
                
                <div class="debug-info">
                    <h4>🔧 Debug Info</h4>
                    <p id="debugInfo">System initialized - Ready for testing</p>
                </div>
                
                <div class="log" id="logContainer">
                    <div class="log-item">
                        <div class="log-time">System Ready</div>
                        <div class="log-message">Criminal Detection System Initialized - FIXED VERSION</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let statsInterval;
        
        // Load location on page load
        window.onload = function() {
            updateLocation();
        };
        
        function startDetection() {
            fetch('/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                        document.getElementById('videoStream').style.display = 'block';
                        document.getElementById('noVideo').style.display = 'none';
                        document.getElementById('status').className = 'status status-active';
                        document.getElementById('statusText').textContent = 'ACTIVE';
                        document.getElementById('statusDetails').textContent = 'AI Detection Running - Monitoring for threats';
                        addLog('SYSTEM STARTED', 'Detection system activated - FIXED VERSION');
                        document.getElementById('debugInfo').textContent = 'Detection loop started - Processing frames every 2nd frame';
                        startStatsUpdate();
                        
                        // Force reload video stream
                        setTimeout(() => {
                            document.getElementById('videoStream').src = '/video_feed?' + new Date().getTime();
                        }, 1000);
                    } else {
                        alert('Failed to start detection: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Failed to start detection: ' + error);
                });
        }
        
        function stopDetection() {
            fetch('/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        document.getElementById('videoStream').style.display = 'none';
                        document.getElementById('noVideo').style.display = 'block';
                        document.getElementById('status').className = 'status status-inactive';
                        document.getElementById('statusText').textContent = 'OFFLINE';
                        document.getElementById('statusDetails').textContent = 'Detection stopped';
                        addLog('SYSTEM STOPPED', 'Detection system deactivated');
                        document.getElementById('debugInfo').textContent = 'System stopped - Ready for restart';
                        stopStatsUpdate();
                    }
                });
        }
        
        function testSMS() {
            document.getElementById('debugInfo').textContent = 'Testing SMS functionality...';
            fetch('/test-sms', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('✅ Test SMS sent successfully!');
                        addLog('SMS TEST', 'Test message sent to configured numbers');
                        document.getElementById('debugInfo').textContent = 'SMS test successful - Twilio working correctly';
                    } else {
                        alert('❌ SMS test failed: ' + data.message);
                        addLog('SMS ERROR', 'Failed to send test SMS: ' + data.message);
                        document.getElementById('debugInfo').textContent = 'SMS test failed - Check console for details';
                    }
                });
        }
        
        function updateLocation() {
            fetch('/location')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('locationText').textContent = 
                            `${data.location.city}, ${data.location.region}, ${data.location.country}`;
                        document.getElementById('coordinatesText').textContent = 
                            `Coordinates: ${data.location.latitude.toFixed(4)}, ${data.location.longitude.toFixed(4)}`;
                        addLog('LOCATION UPDATED', `${data.location.city}, ${data.location.region}`);
                    } else {
                        document.getElementById('locationText').textContent = 'Location unavailable';
                        document.getElementById('coordinatesText').textContent = 'Coordinates: N/A';
                    }
                });
        }
        
        function startStatsUpdate() {
            statsInterval = setInterval(updateStats, 1000);
        }
        
        function stopStatsUpdate() {
            if (statsInterval) clearInterval(statsInterval);
        }
        
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('detectionCount').textContent = data.detections;
                    document.getElementById('alertCount').textContent = data.alerts;
                    
                    if (data.latest_alert) {
                        document.getElementById('status').className = 'status status-alert';
                        document.getElementById('statusDetails').textContent = `🚨 ${data.latest_alert.weapon} DETECTED! Alert sent to authorities`;
                        addLog('🚨 THREAT DETECTED', `${data.latest_alert.weapon} detected - SMS alert sent`);
                        
                        setTimeout(() => {
                            if (document.getElementById('statusText').textContent === 'ACTIVE') {
                                document.getElementById('status').className = 'status status-active';
                                document.getElementById('statusDetails').textContent = 'AI Detection Running - Monitoring for threats';
                            }
                        }, 5000);
                    }
                });
        }
        
        function addLog(title, message) {
            const log = document.getElementById('logContainer');
            const item = document.createElement('div');
            item.className = 'log-item';
            item.innerHTML = `
                <div class="log-time">${new Date().toLocaleTimeString()}</div>
                <div class="log-message">${title}: ${message}</div>
            `;
            log.insertBefore(item, log.firstChild);
            
            // Keep only last 10 entries
            while (log.children.length > 10) {
                log.removeChild(log.lastChild);
            }
        }
    </script>
</body>
</html>
    ''')

@app.route('/start', methods=['POST'])
def start_detection():
    """Start detection system"""
    global detection_thread
    
    if detector.detection_active:
        return jsonify({'success': False, 'message': 'Already running'})
    
    try:
        detector.detection_active = True
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()
        
        print("🚨 Detection system started!")
        return jsonify({'success': True, 'message': 'Detection started'})
        
    except Exception as e:
        print(f"❌ Error starting detection: {e}")
        detector.detection_active = False
        return jsonify({'success': False, 'message': str(e)})

@app.route('/stop', methods=['POST'])
def stop_detection():
    """Stop detection system"""
    detector.detection_active = False
    print("🛑 Detection system stopped!")
    return jsonify({'success': True, 'message': 'Detection stopped'})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get current statistics"""
    latest_alert = None
    if detector.alerts_sent > 0 and (time.time() - detector.last_alert_time) < 10:
        latest_alert = {'weapon': 'WEAPON', 'time': detector.last_alert_time}
    
    return jsonify({
        'detections': detector.total_detections,
        'alerts': detector.alerts_sent,
        'latest_alert': latest_alert
    })

@app.route('/location')
def get_location():
    """Get current location"""
    if detector.current_location:
        return jsonify({
            'success': True,
            'location': detector.current_location
        })
    else:
        return jsonify({'success': False, 'message': 'Location not available'})

@app.route('/test-sms', methods=['POST'])
def test_sms():
    """Test SMS functionality with detailed debugging"""
    print("🧪 SMS TEST INITIATED")
    try:
        # Test with simple message first
        success = detector.send_alert("TEST WEAPON", 95.0)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'SMS sent successfully to {len(detector.police_numbers)} numbers'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'SMS test failed - check console for detailed error messages'
            })
    except Exception as e:
        print(f"❌ Test SMS Exception: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

if __name__ == '__main__':
    print("🚨 CRIMINAL DETECTION SYSTEM")
    print("=" * 50)
    print("📱 SMS: Configure your Twilio credentials in the code")
    print("🎯 Model: Using your trained YOLO model")
    print("📍 Location: Auto-detected via IP geolocation")
    print("💾 Footage: Saved to 'detected_footage' directory")
    print("🌐 Web: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)