#!/usr/bin/env python3
"""
Webcam with guaranteed race display
"""

from deepface import DeepFace
import cv2
import os
import time

def main():
    """
    Launch webcam with guaranteed race display
    """

    # Create database directory
    db_path = "./facial_db"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        print(f"Created database directory: {db_path}")

    print("Launching webcam with GUARANTEED race display...")
    print("Race will be shown in LARGE PINK TEXT above each face")
    print("Press 'q' in the video window to quit")
    print("Available models: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace")

    # Use OpenCV directly for more control over display
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_count = 0
    last_analysis_time = 0
    analysis_interval = 2.0/3.0  # Analyze every ~0.67 seconds (tripled frequency)

    # Store last analysis results for persistence
    last_faces_data = []
    display_duration = 4.0  # Keep displaying results for 4 seconds

    print("Webcam opened! Race detection is ACTIVE.")
    print("Analysis updates every ~0.67 seconds (tripled frequency), results persist for 4 seconds")
    print("Look for PINK 'RACE: [TYPE]' text above faces")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        frame_count += 1

        # Analyze faces periodically
        if current_time - last_analysis_time > analysis_interval:
            try:
                # Detect faces
                faces = DeepFace.extract_faces(frame, detector_backend="ssd", enforce_detection=False)
                last_faces_data = []  # Reset previous data

                for face_data in faces:
                    facial_area = face_data['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                    # Analyze the face region
                    face_img = frame[y:y+h, x:x+w]
                    if face_img.size > 0 and face_img.shape[0] > 10 and face_img.shape[1] > 10:
                        try:
                            analysis = DeepFace.analyze(
                                face_img,
                                actions=['age', 'gender', 'emotion', 'race'],
                                enforce_detection=False,
                                silent=True
                            )[0]

                            # Get confidence levels for all emotions and races
                            emotion_confidences = analysis.get('emotion', {})
                            race_confidences = analysis.get('race', {})

                            # Format confidence strings - show ALL categories
                            emotion_conf_text = "\n".join([f"{k}: {v:.1f}%" for k, v in sorted(emotion_confidences.items(), key=lambda x: x[1], reverse=True)])  # All emotions
                            race_conf_text = "\n".join([f"{k}: {v:.1f}%" for k, v in sorted(race_confidences.items(), key=lambda x: x[1], reverse=True)])  # All races

                            # Store analysis results with timestamp
                            face_info = {
                                'x': x, 'y': y, 'w': w, 'h': h,
                                'age': analysis.get('age', 'Unknown'),
                                'gender': analysis.get('dominant_gender', 'Unknown'),
                                'dominant_emotion': analysis.get('dominant_emotion', 'Unknown'),
                                'dominant_race': analysis.get('dominant_race', 'Unknown'),
                                'emotion_confidences': emotion_conf_text,
                                'race_confidences': race_conf_text,
                                'timestamp': current_time
                            }
                            last_faces_data.append(face_info)

                        except Exception as e:
                            print(f"Analysis error: {e}")

            except Exception as e:
                print(f"Detection error: {e}")

            last_analysis_time = current_time

        # Display stored face data if within display duration
        for face_info in last_faces_data:
            if current_time - face_info['timestamp'] < display_duration:
                x, y, w, h = face_info['x'], face_info['y'], face_info['w'], face_info['h']

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                # Display RACE in LARGE PINK TEXT FIRST
                cv2.putText(frame, f"RACE: {face_info['dominant_race'].upper()}", (x, y-80),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

                # Display race confidences (top 4 for space)
                y_offset = y - 50
                race_lines = face_info['race_confidences'].split('\n')[:4]  # Show top 4 races
                for line in race_lines:
                    cv2.putText(frame, line, (x, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 12

                # Display emotion in YELLOW TEXT
                cv2.putText(frame, f"EMOTION: {face_info['dominant_emotion'].upper()}", (x, y+h+25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Display emotion confidences (top 4 for space)
                y_offset = y + h + 42
                emotion_lines = face_info['emotion_confidences'].split('\n')[:4]  # Show top 4 emotions
                for line in emotion_lines:
                    cv2.putText(frame, line, (x, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 12

                # Display age and gender at bottom
                age_display = int(face_info['age']) if isinstance(face_info['age'], (int, float)) else face_info['age']
                cv2.putText(frame, f"Age: {age_display} | Gender: {face_info['gender']}", (x, y+h+80),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add status text to frame
        cv2.putText(frame, "RACE DETECTION ACTIVE - Updates every ~0.67s", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Show frame
        cv2.imshow('Webcam with Race Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    main()
