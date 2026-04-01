import cv2
import mediapipe as mp
import time
import pyautogui as pgi
from angle_calc import angle_calc
import os
import mimetypes
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

mimetypes.init()
root = Tk()
variable1 = StringVar()
variable2 = StringVar()

# Set up the main window
root.title("PosturaSenseAI - Professional Posture Assessment")
root.geometry("1000x600")
root.configure(bg="#F4F4F4")

# Title Label
title_label = Label(root, text="PosturaSenseAI", font=("Helvetica", 28, "bold"), fg="#333", bg="#F4F4F4")
title_label.pack(pady=20)

# Frame for buttons and status
frame = Frame(root, bg="#FFFFFF", bd=2, relief=RIDGE)
frame.place(relx=0.5, rely=0.5, anchor=CENTER, width=800, height=300)

status_label1 = Label(frame, textvariable=variable1, font=("Helvetica", 16), fg="#2E8B57", bg="#FFFFFF")
status_label1.pack(pady=10)

status_label2 = Label(frame, textvariable=variable2, font=("Helvetica", 16), fg="#2E8B57", bg="#FFFFFF")
status_label2.pack(pady=10)

# Function to display alerts
def display_alert(message, title):
    messagebox.showwarning(title, message)

# Image Pose Estimation
def image_pose_estimation(name):
    img = cv2.imread(name)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    pose1 = []

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            x_y_z = [lm.x, lm.y, lm.z, lm.visibility]
            pose1.append(x_y_z)
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = (0, 128, 255) if id % 2 == 0 else (128, 0, 128)
            cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

    img = cv2.resize(img, (700, 700))
    cv2.imshow("Posture Image", img)

    rula, reba = angle_calc(pose1)
    variable1.set("RULA Score: " + rula)
    variable2.set("REBA Score: " + reba)
    root.update()

    if rula and reba:
        if int(rula) > 3:
            display_alert("Upper body posture not proper!", "Warning")
        elif int(reba) > 4:
            display_alert("Full body posture not proper!", "Warning")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Video Pose Estimation
def video_pose_estimation(name):
    cap = cv2.VideoCapture(name)
    while True:
        success, img = cap.read()
        if not success:
            break
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        pose1 = []

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                x_y_z = [lm.x, lm.y, lm.z, lm.visibility]
                pose1.append(x_y_z)
                cx, cy = int(lm.x * w), int(lm.y * h)
                color = (0, 128, 255) if id % 2 == 0 else (128, 0, 128)
                cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        img = cv2.resize(img, (800, 600))
        cv2.imshow("Posture Video", img)

        rula, reba = angle_calc(pose1)
        variable1.set("RULA Score: " + rula)
        variable2.set("REBA Score: " + reba)
        root.update()

        if (rula != "NULL") and (reba != "NULL"):
            if int(rula) > 3:
                display_alert("Upper body posture not proper!", "Warning")
            if int(reba) > 4:
                display_alert("Full body posture not proper!", "Warning")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Webcam Pose Estimation
def webcam():
    video_pose_estimation(0)

# Browse for file (image or video)
def browsefunc():
    filename = filedialog.askopenfilename()
    mimestart = mimetypes.guess_type(str(filename))[0]

    if mimestart:
        mimestart = mimestart.split('/')[0]

    if mimestart == 'video':
        video_pose_estimation(str(filename))
    elif mimestart == 'image':
        image_pose_estimation(str(filename))

# Stylish buttons
btn1 = Button(frame, text="Analyse from Photo/Video", font=("Helvetica", 16), bg="#000", fg="#FFF", command=browsefunc)
btn1.pack(pady=20, padx=10)

btn2 = Button(frame, text="Live Posture Analysis", font=("Helvetica", 16), bg="#000", fg="#FFF", command=webcam)
btn2.pack(pady=20, padx=10)

# Run the GUI
root.mainloop()
