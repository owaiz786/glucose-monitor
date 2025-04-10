from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from collections import deque
import time
import threading
import os
import json
from datetime import datetime

# Create Flask app
app = Flask(__name__)

# Global variables
frame_lock = threading.Lock()
current_frame = None
current_glucose = None
current_eye_frame = None
glucose_values = []
time_values = []

class ImprovedGlucoseEstimator:
    def __init__(self):
        self.sequence_length = 20
        self.feature_buffer = deque(maxlen=self.sequence_length)
        self.history_buffer = deque(maxlen=100)
        self.glucose_values = []  # Store glucose values for plotting
        self.time_values = []     # Store time values for plotting
        
        # Load eye cascade classifier
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(eye_cascade_path):
            print(f"Warning: Eye cascade file not found at {eye_cascade_path}")
            print("Using default rectangle instead of eye detection.")
            self.eye_cascade = None
        else:
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
        if not os.path.exists(face_cascade_path):
            print(f"Warning: Face cascade file not found at {face_cascade_path}")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Create a figure for plotting
        self.fig = plt.figure(figsize=(8, 3))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_ylim(70, 200)
        self.ax.set_title('Estimated Glucose Level')
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Glucose (mg/dL)')
        self.ax.grid(True)
        plt.tight_layout()
        
        # Starting glucose and trend
        self.base_glucose = 100
        self.trend = 0
        self.start_time = time.time()
        
        # Create a blank image for eye display
        self.eye_display = np.zeros((150, 300), dtype=np.uint8)
    
    def detect_eyes(self, frame):
        """Detect eyes in the frame using Haar cascades"""
        if self.eye_cascade is None or self.face_cascade is None:
            # If cascades not available, just return a default rectangle
            h, w = frame.shape[:2]
            return [(w//4, h//4, w//2, h//2)]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        eyes_list = []
        eye_images = []
        
        # For each face, detect eyes
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Define region of interest for eye detection
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Add eye coordinates to the list
            for (ex, ey, ew, eh) in eyes:
                # Convert to global coordinates
                global_ex = x + ex
                global_ey = y + ey
                
                eyes_list.append((global_ex, global_ey, ew, eh))
                
                # Get the grayscale eye image
                eye_gray = gray[global_ey:global_ey+eh, global_ex:global_ex+ew]
                eye_images.append(eye_gray)
                
                # Draw rectangle around the eye
                cv2.rectangle(frame, (global_ex, global_ey), (global_ex+ew, global_ey+eh), (0, 255, 0), 2)
        
        # Update the eye display window
        self.update_eye_display(eye_images)
        
        # If no eyes detected, return a default rectangle
        if not eyes_list:
            h, w = frame.shape[:2]
            return [(w//4, h//4, w//2, h//2)]
            
        return eyes_list
    
    def update_eye_display(self, eye_images):
        """Create a display of grayscale eye images"""
        if not eye_images:
            # If no eyes detected, show a blank image
            self.eye_display = np.zeros((150, 300), dtype=np.uint8)
            cv2.putText(self.eye_display, "No eyes detected", (50, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return
        
        # Create a blank canvas
        display_height = 150
        display_width = 300
        self.eye_display = np.zeros((display_height, display_width), dtype=np.uint8)
        
        # Limit to max 2 eyes
        eye_images = eye_images[:2]
        
        # Calculate layout
        n_eyes = len(eye_images)
        if n_eyes == 1:
            # Single eye in the center
            eye = eye_images[0]
            
            # Resize to fit
            max_dim = min(display_width, display_height)
            scale = min(max_dim / eye.shape[1], max_dim / eye.shape[0]) * 0.8
            new_width = int(eye.shape[1] * scale)
            new_height = int(eye.shape[0] * scale)
            
            resized_eye = cv2.resize(eye, (new_width, new_height))
            
            # Calculate position to center
            x_offset = (display_width - new_width) // 2
            y_offset = (display_height - new_height) // 2
            
            # Place on canvas
            self.eye_display[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_eye
            
        elif n_eyes == 2:
            # Two eyes side by side
            for i, eye in enumerate(eye_images):
                # Resize to fit
                max_width = display_width // 2
                max_height = display_height
                
                scale = min(max_width / eye.shape[1], max_height / eye.shape[0]) * 0.8
                new_width = int(eye.shape[1] * scale)
                new_height = int(eye.shape[0] * scale)
                
                resized_eye = cv2.resize(eye, (new_width, new_height))
                
                # Calculate position
                x_offset = i * (display_width // 2) + (display_width // 4 - new_width // 2)
                y_offset = (display_height - new_height) // 2
                
                # Place on canvas
                try:
                    self.eye_display[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_eye
                except ValueError as e:
                    print(f"Error placing eye: {e}")
                    print(f"Canvas shape: {self.eye_display.shape}, Image shape: {resized_eye.shape}")
                    print(f"Offsets: x={x_offset}, y={y_offset}, width={new_width}, height={new_height}")
    
    def extract_eye_features(self, frame, eye_coords):
        """Extract features from detected eyes"""
        features = []
        
        for (x, y, w, h) in eye_coords:
            # Extract eye region
            eye_roi = frame[y:y+h, x:x+w]
            
            if eye_roi.size == 0:  # Skip if ROI is empty
                continue
                
            # Convert to grayscale
            gray_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate basic features
            brightness = np.mean(gray_roi) / 255.0
            
            # Extract color features from the sclera (white part of the eye)
            if eye_roi.shape[0] > 0 and eye_roi.shape[1] > 0:
                b, g, r = cv2.split(eye_roi)
                sclera_r = np.mean(r) / 255.0
                sclera_g = np.mean(g) / 255.0
                sclera_b = np.mean(b) / 255.0
            else:
                sclera_r, sclera_g, sclera_b = 0.8, 0.8, 0.8
            
            # Simulate pupil size (in reality would need more advanced processing)
            # For simulation, we'll use the inverse of brightness as a proxy
            pupil_size = 1.0 - brightness
            
            # Position (normalized)
            frame_h, frame_w = frame.shape[:2]
            pos_x = (x + w/2) / frame_w
            pos_y = (y + h/2) / frame_h
            
            # Area (normalized)
            eye_area = (w * h) / (frame_w * frame_h)
            
            features.append([pupil_size, sclera_r, sclera_g, sclera_b, pos_x, pos_y, eye_area])
        
        # If we have features from multiple eyes, take the average
        if features:
            avg_features = np.mean(features, axis=0).tolist()
            return avg_features
        else:
            # Return default features if no valid eyes
            return [0.3, 0.8, 0.8, 0.8, 0.5, 0.5, 0.05]
    
    def generate_plot_image(self):
        """Generate plot as an image"""
        if len(self.glucose_values) > 1:
            self.line.set_data(self.time_values, self.glucose_values)
            self.ax.set_xlim(min(self.time_values), max(self.time_values))
            
            # Render to PNG
            canvas = FigureCanvas(self.fig)
            img_buf = io.BytesIO()
            canvas.print_png(img_buf)
            img_buf.seek(0)
            
            # Convert to base64 for embedding in HTML
            img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
            return img_base64
        return None
    
    def predict_glucose(self, feature_sequence):
        """Simulate glucose prediction"""
        # Extract pupil sizes from all frames in sequence
        pupil_sizes = [features[0] for features in feature_sequence]
        
        # Calculate base effects
        pupil_effect = np.mean(pupil_sizes) * 100  # Scale to glucose range
        
        # Add some realistic variations
        time_effect = 5 * np.sin(time.time() / 300)  # Slow natural variation
        
        # Update trend (random walk)
        self.trend += np.random.normal(0, 0.2)  # Small random changes
        self.trend *= 0.98  # Decay factor to prevent drift
        
        # Calculate final glucose value
        glucose = self.base_glucose + pupil_effect + time_effect + self.trend
        
        # Keep in realistic range
        glucose = np.clip(glucose, 70, 180)
        
        return glucose
    
    def process_frame(self, frame):
        """Process a single video frame"""
        if frame is None:
            return None, None, None
            
        # Make a copy to avoid modifying the original
        processed_frame = frame.copy()
        
        # Detect eyes
        eye_coords = self.detect_eyes(processed_frame)
        
        # Extract features
        feature_vector = self.extract_eye_features(processed_frame, eye_coords)
        
        # Add to feature buffer
        self.feature_buffer.append(feature_vector)
        
        # Make prediction if we have enough data
        if len(self.feature_buffer) == self.sequence_length:
            glucose = self.predict_glucose(list(self.feature_buffer))
            self.history_buffer.append(glucose)
            
            # Store for plotting
            current_time = time.time() - self.start_time
            self.glucose_values.append(glucose)
            self.time_values.append(current_time)
            
            # Determine text color based on glucose level
            if glucose < 70:
                color = (0, 0, 255)  # Red for low
            elif glucose > 140:
                color = (0, 165, 255)  # Orange for high
            else:
                color = (0, 255, 0)  # Green for normal
            
            # Add glucose reading to frame
            cv2.putText(processed_frame, f"Glucose: {glucose:.1f} mg/dL", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            return processed_frame, glucose, self.eye_display
        else:
            # Not enough frames yet
            cv2.putText(processed_frame, f"Collecting data: {len(self.feature_buffer)}/{self.sequence_length}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return processed_frame, None, self.eye_display

# Initialize the glucose estimator
glucose_estimator = ImprovedGlucoseEstimator()

def generate_frames():
    """Generate frames from camera"""
    global current_frame, current_glucose, current_eye_frame
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Process frame
        with frame_lock:
            processed_frame, glucose, eye_frame = glucose_estimator.process_frame(frame)
            current_frame = processed_frame
            current_glucose = glucose
            current_eye_frame = eye_frame
        
        # Small delay to reduce CPU usage
        time.sleep(0.05)
    
    cap.release()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

def generate_video_feed():
    """Generate video feed frames"""
    global current_frame
    while True:
        with frame_lock:
            if current_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small delay to reduce CPU usage
        time.sleep(0.1)

def generate_eye_feed():
    """Generate eye tracking feed frames"""
    global current_eye_frame
    while True:
        with frame_lock:
            if current_eye_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', current_eye_frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small delay to reduce CPU usage
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/eye_feed')
def eye_feed():
    """Eye tracking video streaming route"""
    return Response(generate_eye_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/glucose_data')
def glucose_data():
    """Return current glucose data as JSON"""
    global current_glucose
    with frame_lock:
        data = {
            'glucose': current_glucose if current_glucose is not None else 0,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
    return jsonify(data)

@app.route('/glucose_chart')
def glucose_chart():
    """Return glucose chart data as JSON"""
    with frame_lock:
        data = {
            'times': glucose_estimator.time_values,
            'values': glucose_estimator.glucose_values
        }
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
    camera_thread = threading.Thread(target=generate_frames)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Run Flask app
    app.run(debug=False, threaded=True)
