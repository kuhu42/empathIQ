import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points using the law of cosines."""
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint (joint)
    c = np.array(c)  # Third point

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid invalid values
    return np.degrees(angle)

def determine_faulty_joints(joint_angles):
    """
    Determine faulty joints based on computed joint angles.
    """
    acceptable_ranges = {
        "neck": (160, 180),
        "left_shoulder": (30, 60),
        "right_shoulder": (30, 60),
        "spine": (150, 170),
        "left_elbow": (30, 160),
        "right_elbow": (30, 160)
    }

    alert_messages = {
        "neck": "Adjust your neck alignment.",
        "left_shoulder": "Straighten your left shoulder.",
        "right_shoulder": "Straighten your right shoulder.",
        "spine": "Improve your spine posture.",
        "left_elbow": "Check your left elbow position.",
        "right_elbow": "Check your right elbow position."
    }

    faulty_joints = {}
    for joint, angle in joint_angles.items():
        if joint in acceptable_ranges:
            lower_bound, upper_bound = acceptable_ranges[joint]
            if angle < lower_bound or angle > upper_bound:
                faulty_joints[joint] = alert_messages.get(joint, f"Adjust your {joint} posture.")

    return faulty_joints

def get_faulty_joint_summary(joint_angles):
    """Calculate final posture score and detect faulty joints."""
    faulty_joints = determine_faulty_joints(joint_angles)
    final_score = 100 - (len(faulty_joints) * 15)  # Deduct 15 points per faulty joint
    final_score = max(0, final_score)
    return final_score, list(faulty_joints.keys())

joint_angles = {}
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            neck = [(shoulder_left[0] + shoulder_right[0]) / 2,
                    (shoulder_left[1] + shoulder_right[1]) / 2]
            
            joint_angles["left_elbow"] = calculate_angle(shoulder_left, elbow_left, wrist_left)
            joint_angles["right_elbow"] = calculate_angle(shoulder_right, elbow_right, wrist_right)
            joint_angles["left_shoulder"] = calculate_angle(neck, shoulder_left, elbow_left)
            joint_angles["right_shoulder"] = calculate_angle(neck, shoulder_right, elbow_right)
            joint_angles["spine"] = calculate_angle(shoulder_left, hip, [hip[0], hip[1] - 0.1])
            joint_angles["neck"] = calculate_angle(shoulder_left, neck, shoulder_right)
            
            faulty_joints = determine_faulty_joints(joint_angles)
            final_score, faulty_joint_names = get_faulty_joint_summary(joint_angles)
            
            for i, (joint, alert) in enumerate(faulty_joints.items()):
                cv2.putText(frame, f"{joint}: {alert}", (50, 80 + (i * 30)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.putText(frame, f'Posture Score: {final_score}', (50, 120 + (len(faulty_joints) * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Faulty Joints: {", ".join(faulty_joint_names)}', (50, 150 + (len(faulty_joints) * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

        cv2.imshow('Posture Analysis', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
