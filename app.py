import os
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def analyze_video(video_path):
    """
    Analyze video for potential deepfake indicators using MediaPipe face mesh
    Returns confidence score and analysis results
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    inconsistency_score = 0
    landmark_distances = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate facial feature consistency
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            
            # Calculate distances between key facial features
            nose_tip = landmarks[4]
            left_eye = landmarks[133]
            right_eye = landmarks[362]
            mouth_center = landmarks[0]
            
            # Calculate facial symmetry and proportions
            eye_distance = np.linalg.norm(left_eye - right_eye)
            nose_mouth_distance = np.linalg.norm(nose_tip - mouth_center)
            ratio = eye_distance / nose_mouth_distance
            landmark_distances.append(ratio)
            
            # Check for unnatural movements and artifacts
            blurriness = cv2.Laplacian(frame, cv2.CV_64F).var()
            if blurriness < 100 or ratio < 0.3 or ratio > 0.7:
                inconsistency_score += 1
        
        frame_count += 1
        if frame_count > 100:  # Analyze first 100 frames only
            break
    
    face_mesh.close()
    cap.release()
    
    if frame_count == 0:
        return {
            "is_deepfake": False,
            "confidence": 0,
            "message": "Could not analyze video - no frames found"
        }
    
    confidence = (inconsistency_score / frame_count) * 100
    is_deepfake = confidence > 50  # Arbitrary threshold
    
    return {
        "is_deepfake": is_deepfake,
        "confidence": round(confidence, 2),
        "message": "Video analysis complete"
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
        
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
        
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return jsonify({'error': 'Invalid file format'}), 400
        
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)
    
    try:
        result = analyze_video(video_path)
        os.remove(video_path)  # Clean up uploaded file
        return jsonify(result)
    except Exception as e:
        os.remove(video_path)  # Clean up uploaded file
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)
