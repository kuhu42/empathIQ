import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from deepface import DeepFace
from flask import Flask, Response, jsonify, render_template
import threading
import math
import time
import urllib.request
import os

app = Flask(__name__)

# ── Download MediaPipe pose model if needed ─────────────────────
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe pose model (~5MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded ✅")

# ── MediaPipe Pose (Tasks API — works with mediapipe 0.10.x) ────
_pose_result_cache = None
_pose_result_lock = threading.Lock()

def _result_callback(result, output_image, timestamp_ms):
    global _pose_result_cache
    with _pose_result_lock:
        _pose_result_cache = result

options = PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=_result_callback,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose_landmarker = PoseLandmarker.create_from_options(options)

# ── Drawing helper ───────────────────────────────────────────────
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (0,11),(0,12),(7,8),
]

def draw_pose(frame, landmarks, h, w):
    pts = {}
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts[i] = (cx, cy)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 120), -1)
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 180, 255), 2)


# ── Shared state ────────────────────────────────────────────────
state = {
    "emotion": "unknown",
    "emotion_scores": {},
    "posture": "unknown",
    "posture_features": {},
    "behavior": "unknown",
    "behavior_confidence": 0.0,
    "behavior_detail": "",
    "fps": 0
}
state_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()


# ── Posture Analysis ────────────────────────────────────────────
def extract_posture_features(landmarks):
    """Extract posture features from MediaPipe Tasks API landmark list."""

    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    nose        = pt(0)
    l_shoulder  = pt(11);  r_shoulder = pt(12)
    l_hip       = pt(23);  r_hip      = pt(24)
    l_ear       = pt(7);   r_ear      = pt(8)
    l_elbow     = pt(13);  r_elbow    = pt(14)
    l_wrist     = pt(15);  r_wrist    = pt(16)

    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip      = (l_hip + r_hip) / 2

    shoulder_tilt  = abs(l_shoulder[1] - r_shoulder[1])
    head_forward   = nose[0] - mid_shoulder[0]
    spine_vec      = mid_shoulder - mid_hip
    spine_angle    = math.degrees(math.atan2(abs(spine_vec[0]), abs(spine_vec[1]) + 1e-6))
    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
    avg_wrist_y    = (l_wrist[1] + r_wrist[1]) / 2
    arm_raise      = mid_shoulder[1] - avg_wrist_y
    head_tilt      = abs(l_ear[1] - r_ear[1])
    elbow_spread   = abs(l_elbow[0] - r_elbow[0])
    openness       = elbow_spread / (shoulder_width + 1e-6)
    slouch_score   = nose[1] - mid_shoulder[1]

    features = {
        "shoulder_tilt":  float(shoulder_tilt),
        "head_forward":   float(head_forward),
        "spine_angle":    float(spine_angle),
        "shoulder_width": float(shoulder_width),
        "arm_raise":      float(arm_raise),
        "head_tilt":      float(head_tilt),
        "openness":       float(openness),
        "slouch_score":   float(slouch_score),
    }

    if spine_angle > 15:
        posture_label = "leaning"
    elif slouch_score > 0.15:
        posture_label = "slouched"
    elif arm_raise > 0.1:
        posture_label = "arms_raised"
    elif openness > 1.2:
        posture_label = "open"
    elif openness < 0.6:
        posture_label = "closed"
    else:
        posture_label = "neutral"

    return posture_label, features


# ── Behavior Inference ──────────────────────────────────────────
BEHAVIOR_RULES = [
    ("Engaged & Attentive",    "Upright posture, positive affect",
     lambda e, p, f: e in ("happy", "surprise") and p in ("neutral", "open")),
    ("Stressed / Anxious",     "Tense posture, fearful expression",
     lambda e, p, f: e in ("fear", "angry") and p in ("closed", "slouched")),
    ("Disengaged / Bored",     "Slouching, neutral expression",
     lambda e, p, f: e == "neutral" and p in ("slouched", "leaning")),
    ("Confident & Expressive", "Open body language, expressive face",
     lambda e, p, f: p == "open" and e in ("happy", "surprise", "neutral")),
    ("Defensive",              "Closed posture, negative emotion",
     lambda e, p, f: p == "closed" and e in ("angry", "disgust", "fear")),
    ("Excited / Animated",     "Arms active, strong positive emotion",
     lambda e, p, f: f.get("arm_raise", 0) > 0.08 and e in ("happy", "surprise")),
    ("Sad / Withdrawn",        "Slouched, sad expression",
     lambda e, p, f: e in ("sad", "disgust") and p in ("slouched", "closed", "neutral")),
    ("Thinking / Reflective",  "Head tilted, neutral face",
     lambda e, p, f: f.get("head_tilt", 0) > 0.04 and e == "neutral"),
    ("Frustrated",             "Leaning forward, angry expression",
     lambda e, p, f: e == "angry" and p in ("leaning", "open")),
    ("Calm & Relaxed",         "Balanced posture, neutral/happy affect",
     lambda e, p, f: e in ("neutral", "happy") and p == "neutral"),
]

def infer_behavior(emotion, posture, features):
    matches = []
    for label, detail, cond in BEHAVIOR_RULES:
        try:
            if cond(emotion, posture, features):
                matches.append((label, detail))
        except Exception:
            pass
    if not matches:
        return "Observing...", 0.4, "Gathering more data"
    label, detail = matches[0]
    confidence = min(0.5 + 0.1 * len(matches), 0.95)
    return label, confidence, detail


# ── Main Processing Loop ────────────────────────────────────────
def process_frames():
    global latest_frame, state

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    emotion_result  = {"emotion": "unknown", "scores": {}}
    emotion_counter = 0
    EMOTION_INTERVAL = 5  # run DeepFace every N frames (it's slow)

    prev_time = time.time()
    frame_ts  = 0  # monotonically increasing ms timestamp for MediaPipe

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Pose Detection (async live-stream) ──
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        pose_landmarker.detect_async(mp_image, frame_ts)

        posture_label    = "unknown"
        posture_features = {}

        with _pose_result_lock:
            cached = _pose_result_cache

        if cached and cached.pose_landmarks:
            lms = cached.pose_landmarks[0]
            posture_label, posture_features = extract_posture_features(lms)
            draw_pose(frame, lms, h, w)

        # ── Emotion Detection (every N frames) ──
        emotion_counter += 1
        if emotion_counter >= EMOTION_INTERVAL:
            emotion_counter = 0
            try:
                result = DeepFace.analyze(
                    rgb,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(result, list):
                    result = result[0]
                emotion_result = {
                    "emotion": result["dominant_emotion"],
                    "scores":  {k: round(v, 1) for k, v in result["emotion"].items()}
                }
                if "region" in result:
                    r = result["region"]
                    cv2.rectangle(frame,
                                  (r["x"], r["y"]),
                                  (r["x"]+r["w"], r["y"]+r["h"]),
                                  (0, 255, 120), 2)
            except Exception:
                pass

        # ── Behavior Inference ──
        behavior, confidence, detail = infer_behavior(
            emotion_result["emotion"], posture_label, posture_features
        )

        # ── FPS ──
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        # ── Overlay ──
        cv2.putText(frame, f"Emotion: {emotion_result['emotion']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
        cv2.putText(frame, f"Posture: {posture_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
        cv2.putText(frame, f"Behavior: {behavior}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # ── Update shared state ──
        with state_lock:
            state.update({
                "emotion":              emotion_result["emotion"],
                "emotion_scores":       emotion_result["scores"],
                "posture":              posture_label,
                "posture_features":     {k: round(v, 3) for k, v in posture_features.items()},
                "behavior":             behavior,
                "behavior_confidence":  round(confidence, 2),
                "behavior_detail":      detail,
                "fps":                  round(fps, 1)
            })

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = buffer.tobytes()

    cap.release()


# ── Flask Routes ────────────────────────────────────────────────
def gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/state")
def get_state():
    with state_lock:
        return jsonify(state)


if __name__ == "__main__":
    t = threading.Thread(target=process_frames, daemon=True)
    t.start()
    print("🚀 Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)