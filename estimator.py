import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import pandas as pd
from datetime import datetime
import os
import cv2
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
        
        # Create a separate figure for plotting
        plt.figure(figsize=(10, 4))
        self.fig = plt.figure(figsize=(10, 4))
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
    
    def update_plot(self):
        """Update the glucose trend plot in a separate window"""
        if len(self.glucose_values) > 1:
            self.line.set_data(self.time_values, self.glucose_values)
            self.ax.set_xlim(min(self.time_values), max(self.time_values))
            
            # Force redraw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Short pause to allow plot to update
    
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
        # Detect eyes
        eye_coords = self.detect_eyes(frame)
        
        # Extract features
        feature_vector = self.extract_eye_features(frame, eye_coords)
        
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
            cv2.putText(frame, f"Glucose: {glucose:.1f} mg/dL", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            return frame, glucose
        else:
            # Not enough frames yet
            cv2.putText(frame, f"Collecting data: {len(self.feature_buffer)}/{self.sequence_length}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame, None
    
    def run(self, video_source=0):
        """Run real-time glucose estimation simulation"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # For saving results
        results = []
        self.start_time = time.time()
        
        # Create windows
        cv2.namedWindow('Contactless Glucose Monitoring (Simulation)')
        cv2.namedWindow('Eye Tracking (Grayscale)')
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame, glucose = self.process_frame(frame)
                
                # Show the main frame
                cv2.imshow('Contactless Glucose Monitoring (Simulation)', processed_frame)
                
                # Show the eye display
                cv2.imshow('Eye Tracking (Grayscale)', self.eye_display)
                
                # Update plot if we have a new glucose reading
                if glucose is not None:
                    # Record result
                    elapsed_time = time.time() - self.start_time
                    results.append({
                        'time': elapsed_time,
                        'glucose': glucose
                    })
                    
                    # Update plot every 5 readings to avoid excessive redrawing
                    if len(results) % 5 == 0:
                        self.update_plot()
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            
            # Final plot update
            self.update_plot()
            
            # Keep plot window open
            plt.show()
            
            # Save results to CSV
            if results:
                results_df = pd.DataFrame(results)
                filename = f'glucose_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                results_df.to_csv(filename, index=False)
                print(f"Results saved to CSV file: {filename}")

# Run the application
if __name__ == "__main__":
    estimator = ImprovedGlucoseEstimator()
    estimator.run()