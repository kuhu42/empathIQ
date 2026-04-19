# Behavior Predictor

Real-time facial emotion + posture → behavior prediction. No model training required to run.

## Stack

| Component | Library | Training needed? |
|-----------|---------|-----------------|
| Emotion detection | DeepFace (pretrained) | ❌ None |
| Pose estimation | MediaPipe (pretrained) | ❌ None |
| Behavior inference | Rule-based | ❌ None |
| Optional ML upgrade | sklearn GBT | ✅ ~1 min on Colab |

## Setup

```bash
cd behavior_predictor
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## Files

```
behavior_predictor/
├── app.py               ← Main Flask server + all inference logic
├── requirements.txt
├── train_colab.ipynb    ← Optional: train sklearn classifier on Colab GPU
└── templates/
    └── index.html       ← Minimal dark UI
```

## Behaviors Detected

- Engaged & Attentive
- Stressed / Anxious
- Disengaged / Bored
- Confident & Expressive
- Defensive
- Excited / Animated
- Sad / Withdrawn
- Thinking / Reflective
- Frustrated
- Calm & Relaxed

## Optional ML Upgrade (Colab)

1. Open `train_colab.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells (~1 min)
4. Download `behavior_clf.joblib` + `label_encoder.joblib`
5. Place them in project root
6. Follow the code snippet at the bottom of the notebook to swap into `app.py`

## Notes

- DeepFace runs every 5 frames (configurable via `EMOTION_INTERVAL` in app.py)
- MediaPipe runs every frame
- Webcam index 0 assumed — change `cv2.VideoCapture(0)` if needed
- First run downloads DeepFace weights automatically (~100MB)
