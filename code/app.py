from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pyngrok import ngrok
import os

app = Flask(__name__)

# ------------------------------
# Load emotion detection model
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "emotion_model_save.keras")
model = load_model(MODEL_PATH)
emotion_map = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprise"}

# ------------------------------
# Load face detector
# ------------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------------------
# Initialize webcam
# ------------------------------
camera = cv2.VideoCapture(0)

# ------------------------------
# Generate video frames
# ------------------------------
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue  # retry if camera fails

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1)

            pred = model.predict(face, verbose=0)[0]
            emotion = emotion_map[np.argmax(pred)]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show probabilities
            for i, (emo, prob) in enumerate(zip(emotion_map.values(), pred)):
                cv2.putText(frame, f"{emo}: {prob*100:.2f}%",
                            (10, 30 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ------------------------------
# Flask routes
# ------------------------------
@app.route('/')
def index():
    return render_template("index.html")  # Make sure index.html exists

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)