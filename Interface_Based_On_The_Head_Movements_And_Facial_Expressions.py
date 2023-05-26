# -*- coding: utf-8 -*-
import Tkinter as tk
import numpy as np
import cv2
import dlib
import imutils
import pyautogui
from scipy.spatial import distance as dist
from pynput.mouse import Button, Controller
from win32api import GetSystemMetrics

mouse = Controller()
pyautogui.FAILSAFE = True

dlib_face_detector = dlib.get_frontal_face_detector()
dlib_facial_landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
dlib_right_eye_landmarks = list(range(42,48))
dlib_left_eye_landmarks = list(range(36,42))
dlib_right_eyebrow_landmarks = list(range(22,27))
dlib_left_eyebrow_landmarks = list(range(17,22))
dlib_mouth_out_landmarks = list(range(48,61))
dlib_mouth_in_landmarks = list(range(61,68))

def eyebrow_aspect_ratio(eyebrow,eye):
    A = dist.euclidean(eyebrow[1], eye[5])
    B = dist.euclidean(eyebrow[2], eye[4])
    C = dist.euclidean(eyebrow[0], eye[3])
    ear = (A+B)/(2.0*C)
    return ear
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A+B)/(2.0*C)
    return ear
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[5])
    B = dist.euclidean(mouth[2], mouth[4])
    C = dist.euclidean(mouth[0], mouth[3])
    mar = (A+B)/(2.0*C)
    return mar

main_window = tk.Tk()
main_window.title("Środowisko badawcze do testowania interakcji użytkownika z wykorzystaniem ruchów głowy i mimiki")
main_window.geometry("880x170")

description_label = tk.Label(main_window, text="Środowisko badawcze do testowania interakcji użytkownika z wykorzystaniem ruchów głowy i mimiki")
description_label.place(x=90, y=0)
description_label = tk.Label(main_window, text="Wybierz opcje testowania")
description_label.place(x=330, y=20)

description_label  = tk.Label(main_window, text="Prędkość kursora (zalecana od 5 do 30)")
description_label .place(x=10, y=20)

sensitivity = tk.Entry(main_window)
sensitivity.place(x=10, y=55)

option1_button = tk.Button(main_window, text="1. Mimika twarzy", width=21, command=lambda: option1_test())
option1_button.place(x=330, y=55)
option2_button = tk.Button(main_window, text="2. Sterowanie kursorem", width=21, command=lambda: option2_test())
option2_button.place(x=330, y=90)
option3_button = tk.Button(main_window, text="3. Symulator myszy", width=21, command=lambda: option3_test())
option3_button.place(x=330, y=125)


def option1_test():
    option1_test_window = tk.Tk()
    option1_test_window.title("Środowisko badawcze do testowania interakcji użytkownika z wykorzystaniem ruchów głowy i mimiki")
    option1_test_window.geometry("800x400")
    description_label = tk.Label(option1_test_window, text="Ocena poziomu trudności wykonania ruchów mimcznych w celu kliknięcia myszą")
    description_label.place(x=140, y=0)
    description_label = tk.Label(option1_test_window, text="Wybierz opcje testowania")
    description_label.place(x=290, y=25)
    option1_button = tk.Button(option1_test_window, text="1. Uniesienie obu brwi", width=30, command=lambda: option1_test_1())
    option1_button.place(x=250, y=65)
    option2_button = tk.Button(option1_test_window, text="2. Uniesienie lewej brwi", width=30, command=lambda: option1_test_2())
    option2_button.place(x=250, y=105)
    option3_button = tk.Button(option1_test_window, text="3. Uniesienie prawej brwi", width=30, command=lambda: option1_test_3())
    option3_button.place(x=250, y=145)
    option1_button = tk.Button(option1_test_window, text="4. Uśmiechnięcie się", width=30, command=lambda: option1_test_4())
    option1_button.place(x=250, y=185)
    option2_button = tk.Button(option1_test_window, text="5. Otwarcie buzi", width=30, command=lambda: option1_test_5())
    option2_button.place(x=250, y=225)
    option3_button = tk.Button(option1_test_window, text="6. Zamknięcie oczu", width=30, command=lambda: option1_test_6())
    option3_button.place(x=250, y=265)
    option1_button = tk.Button(option1_test_window, text="7. Mrugnięcie lewym okiem", width=30, command=lambda: option1_test_7())
    option1_button.place(x=250, y=305)
    option2_button = tk.Button(option1_test_window, text="8. Mrugnięcie prawym okiem", width=30, command=lambda: option1_test_8())
    option2_button.place(x=250, y=345)
def option1_test_1():
    eyebrow_ar_thresh = 0.75
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                eyebrow_left_ear = eyebrow_aspect_ratio(left_eyebrow_landmarks, left_eye_landmarks)
                eyebrow_right_ear = eyebrow_aspect_ratio(right_eyebrow_landmarks, right_eye_landmarks)
                ear = (eyebrow_left_ear + eyebrow_right_ear) / 2.0

                if ear < eyebrow_ar_thresh:
                    counter_eyebrows += 1
                else:
                    if counter_eyebrows >= eyebrow_ar_consec_frames:
                        total_eyebrows += 1
                    counter_eyebrows = 0

                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)
            cv2.putText(webcam_image_gray, "Licznik: {}".format(total_eyebrows), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option1_test_2():
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                A = dist.euclidean(left_eyebrow_landmarks[3], left_eye_landmarks[3])
                B = dist.euclidean(right_eyebrow_landmarks[3], right_eye_landmarks[3])

                if (A > 13 and B < 13):
                    counter_eyebrows += 1
                else:
                    if counter_eyebrows >= eyebrow_ar_consec_frames:
                        total_eyebrows += 1
                    counter_eyebrows = 0

            cv2.putText(webcam_image_gray, "Licznik: {}".format(total_eyebrows), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for (i, p) in enumerate(left_eyebrow_landmarks):
                left_eyebrow_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option1_test_3():
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                A = dist.euclidean(left_eyebrow_landmarks[3], left_eye_landmarks[3])
                B = dist.euclidean(right_eyebrow_landmarks[3], right_eye_landmarks[3])

                if (B > 13 and A < 13):
                    counter_eyebrows += 1
                else:
                    if counter_eyebrows >= eyebrow_ar_consec_frames:
                        total_eyebrows += 1
                    counter_eyebrows = 0

            cv2.putText(webcam_image_gray, "Licznik: {}".format(total_eyebrows), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for (i, p) in enumerate(right_eyebrow_landmarks):
                right_eyebrow_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option1_test_4():
    mouth_ar_thresh = 31
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            if len(detected_faces) == 1:
                A = dist.euclidean(mouth_out_landmarks[0], mouth_out_landmarks[6])
                if A < mouth_ar_thresh:
                    counter_mouth += 1
                else:
                    if counter_mouth >= mouth_ar_consec_frames:
                        total_mouth += 1
                    counter_mouth = 0

            cv2.putText(webcam_image_gray, "Licznik : {}".format(total_mouth), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for (i, p) in enumerate(mouth_out_landmarks):
                mouth_out_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

            for (i, p) in enumerate(mouth_in_landmarks):
                mouth_in_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option1_test_5():
    mouth_ar_thresh = 0.2
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            if len(detected_faces) == 1:
                mouth_in_ear = mouth_aspect_ratio(mouth_in_landmarks)

                if mouth_in_ear < mouth_ar_thresh:
                    counter_mouth += 1
                else:
                    if counter_mouth >= mouth_ar_consec_frames:
                        total_mouth += 1
                    counter_mouth = 0

            cv2.putText(webcam_image_gray, "Licznik : {}".format(total_mouth), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for (i, p) in enumerate(mouth_out_landmarks):
                mouth_out_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

            for (i, p) in enumerate(mouth_in_landmarks):
                mouth_in_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option1_test_6():
    eye_ar_thresh = 0.25
    eye_ar_consec_frames = 3
    counter_eyes = 0
    total_eyes_blinks = 0
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
            if len(detected_faces) == 1:
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)
                ear = (left_eye_ear + right_eye_ear) / 2.0

                if ear < eye_ar_thresh:
                    counter_eyes += 1
                else:
                    if counter_eyes >= eye_ar_consec_frames:
                        total_eyes_blinks += 1
                    counter_eyes = 0

                cv2.putText(webcam_image_gray, "Licznik : {}".format(total_eyes_blinks), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for (i, p) in enumerate(left_eye_landmarks):
                left_eye_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, left_eye_position, 1, (0, 0, 0), -1)

            for (i, p) in enumerate(right_eye_landmarks):
                right_eye_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, right_eye_position, 1, (0, 0, 0), -1)

            cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option1_test_7():
    eye_ar_thresh = 0.3
    eye_ar_consec_frames = 3
    counter_eyes = 0
    total_eyes_blinks = 0
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
            if len(detected_faces) == 1:
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)

                if left_eye_ear < eye_ar_thresh - 0.07 and right_eye_ear > eye_ar_thresh - 0.11:
                    counter_eyes += 1
                else:
                    if counter_eyes >= eye_ar_consec_frames:
                        total_eyes_blinks += 1
                    counter_eyes = 0
                    cv2.putText(webcam_image_gray, "Licznik : {}".format(total_eyes_blinks), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for (i, p) in enumerate(left_eye_landmarks):
                left_eye_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, left_eye_position, 1, (0, 0, 0), -1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option1_test_8():
    eye_ar_thresh = 0.3
    eye_ar_consec_frames = 3
    counter_eyes = 0
    total_eyes_blinks = 0
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
            if len(detected_faces) == 1:
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)

                if right_eye_ear < eye_ar_thresh - 0.07 and left_eye_ear > eye_ar_thresh - 0.11:
                    counter_eyes += 1
                else:
                    if counter_eyes >= eye_ar_consec_frames:
                        total_eyes_blinks += 1
                    counter_eyes = 0

                cv2.putText(webcam_image_gray, "Licznik : {}".format(total_eyes_blinks), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for (i, p) in enumerate(right_eye_landmarks):
                right_eye_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, right_eye_position, 1, (0, 0, 0), -1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()

def option2_test():
    option2_test_window = tk.Tk()
    option2_test_window.title("Środowisko badawcze do testowania interakcji użytkownika z wykorzystaniem ruchów głowy i mimiki")
    option2_test_window.geometry("880x170")
    description_label = tk.Label(option2_test_window,text="Ocena poziomu trudności sterowania kursorem z wykorzystaniem ruchów głowy")
    description_label.place(x=150, y=0)
    description_label = tk.Label(option2_test_window, text="Wybierz opcje testowania")
    description_label.place(x=340, y=20)
    option1_button = tk.Button(option2_test_window, text="1. Sterowanie bezwzględne", width=30, command=lambda: option2_test_1())
    option1_button.place(x=300, y=55)
    option2_button = tk.Button(option2_test_window, text="2. Sterowanie względne", width=30, command=lambda: option2_test_2())
    option2_button.place(x=300, y=90)
    option3_button = tk.Button(option2_test_window, text="3. Sterowanie proste", width=30, command=lambda: option2_test_3())
    option3_button.place(x=300, y=125)
def option2_test_1():
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            if len(detected_faces) == 1:
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if ((nose_y > 60) and (nose_y < 130)):
                    if ((nose_x > 80) and (nose_x < 160)):
                        if ((nose_x > 130) and (nose_x < 160)):
                            mouse.move(speed, 0)
                        if ((nose_x < 110) and (nose_x > 80)):
                            mouse.move(-speed, 0)
                        if ((nose_y > 60) and (nose_y < 90)):
                            mouse.move(0, -speed)
                        if ((nose_y < 140) and (nose_y > 110)):
                            mouse.move(0, speed)

                for (i, detected_faces) in enumerate(detected_faces):
                    cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y),
                               1, (0, 0, 0), 2)

        cv2.line(webcam_image_gray, (80, 100), (160, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 60), (120, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (80, 60), (160, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 2)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option2_test_2():
    sx = GetSystemMetrics(0)
    sy = GetSystemMetrics(1)
    (camx,camy)=(320,240)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
            nose_x = landmarks_coordinates.part(30).x
            nose_y = landmarks_coordinates.part(30).y

            if len(detected_faces) == 1:
                pyautogui.moveTo(sx - ((nose_x * sx / camx) * 2), ((nose_y * sy / camy)))

                for (i, detected_faces) in enumerate(detected_faces):
                    cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,(0, 0, 0), 2)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option2_test_3():
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
            nose_x = landmarks_coordinates.part(30).x
            nose_y = landmarks_coordinates.part(30).y
            if len(detected_faces) == 1:
                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                for (i, detected_faces) in enumerate(detected_faces):
                    cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1, (0, 0, 0), 2)

        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()

def option3_test():
    option3_test_window = tk.Tk()
    option3_test_window.title("Środowisko badawcze do testowania interakcji użytkownika z wykorzystaniem ruchów głowy i mimiki")
    option3_test_window.geometry("1400x800")
    description_label = tk.Label(option3_test_window, text="Wybierz opcje myszy")
    description_label.place(x=550, y=0)

    option1_0_label = tk.Label(option3_test_window, text="OPCJA NR. 1")
    option1_0_label.place(x=80, y=50)
    option1_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - uniesiona lewa brew")
    option1_1_label.place(x=10, y=75)
    option1_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - uniesiona prawa brew")
    option1_2_label.place(x=10, y=100)
    option1_3_label = tk.Label(option3_test_window, text="Sterowanie - proste")
    option1_3_label.place(x=10, y=125)
    option1_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_1())
    option1_button.place(x=100, y=155)

    option2_0_label = tk.Label(option3_test_window, text="OPCJA NR. 2")
    option2_0_label.place(x=80, y=195)
    option2_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - uniesione brwi")
    option2_1_label.place(x=10, y=220)
    option2_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - otwarte usta")
    option2_2_label.place(x=10, y=245)
    option2_3_label = tk.Label(option3_test_window, text="Sterowanie - proste")
    option2_3_label.place(x=10, y=270)
    option2_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_2())
    option2_button.place(x=100, y=300)

    option3_0_label = tk.Label(option3_test_window, text="OPCJA NR. 3")
    option3_0_label.place(x=80, y=340)
    option3_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - uniesione brwi")
    option3_1_label.place(x=10, y=360)
    option3_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - uśmiech")
    option3_2_label.place(x=10, y=385)
    option3_3_label = tk.Label(option3_test_window, text="Sterowanie - proste")
    option3_3_label.place(x=10, y=410)
    option3_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_3())
    option3_button.place(x=100, y=450)

    option4_0_label = tk.Label(option3_test_window, text="OPCJA NR. 4")
    option4_0_label.place(x=80, y=490)
    option4_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - otwarte usta")
    option4_1_label.place(x=10, y=515)
    option4_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - zamkniete oczy")
    option4_2_label.place(x=10, y=540)
    option4_3_label = tk.Label(option3_test_window, text="Sterowanie - proste")
    option4_3_label.place(x=10, y=565)
    option4_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_4())
    option4_button.place(x=100, y=605)

    option5_0_label = tk.Label(option3_test_window, text="OPCJA NR. 5")
    option5_0_label.place(x=80, y=645)
    option5_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - zamkniete lewe oko")
    option5_1_label.place(x=10, y=670)
    option5_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - zamkniete prawe oko")
    option5_2_label.place(x=10, y=695)
    option5_3_label = tk.Label(option3_test_window, text="Sterowanie - proste")
    option5_3_label.place(x=10, y=720)
    option5_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_5())
    option5_button.place(x=100, y=750)

    option6_0_label = tk.Label(option3_test_window, text="OPCJA NR. 6")
    option6_0_label.place(x=400, y=45)
    option6_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - uniesiona lewa brew")
    option6_1_label.place(x=330, y=70)
    option6_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - uniesiona prawa brew")
    option6_2_label.place(x=330, y=95)
    option6_3_label = tk.Label(option3_test_window, text="Sterowanie - bezwzgledne")
    option6_3_label.place(x=330, y=120)
    option6_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_6())
    option6_button.place(x=420, y=150)

    option7_0_label = tk.Label(option3_test_window, text="OPCJA NR. 7")
    option7_0_label.place(x=400, y=190)
    option7_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - uniesione brwi")
    option7_1_label.place(x=330, y=215)
    option7_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - otwarte usta")
    option7_2_label.place(x=330, y=240)
    option7_3_label = tk.Label(option3_test_window, text="Sterowanie - bezwzgledne")
    option7_3_label.place(x=330, y=265)
    option7_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_7())
    option7_button.place(x=420, y=295)

    option8_0_label = tk.Label(option3_test_window, text="OPCJA NR. 8")
    option8_0_label.place(x=400, y=335)
    option8_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - uniesione brwi")
    option8_1_label.place(x=330, y=360)
    option8_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - uśmiech")
    option8_2_label.place(x=330, y=385)
    option8_3_label = tk.Label(option3_test_window, text="Sterowanie - bezwzgledne")
    option8_3_label.place(x=330, y=410)
    option8_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_8())
    option8_button.place(x=420, y=440)

    option9_0_label = tk.Label(option3_test_window, text="OPCJA NR. 9")
    option9_0_label.place(x=400, y=490)
    option9_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - otwarte usta")
    option9_1_label.place(x=330, y=515)
    option9_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - zamkniete oczy")
    option9_2_label.place(x=330, y=540)
    option9_3_label = tk.Label(option3_test_window, text="Sterowanie - bezwzgledne")
    option9_3_label.place(x=330, y=565)
    option9_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_9())
    option9_button.place(x=420, y=595)

    option10_0_label = tk.Label(option3_test_window, text="OPCJA NR. 10")
    option10_0_label.place(x=400, y=645)
    option10_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - zamkniete lewe oko")
    option10_1_label.place(x=330, y=670)
    option10_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - zamkniete prawe oko")
    option10_2_label.place(x=330, y=695)
    option10_3_label = tk.Label(option3_test_window, text="Sterowanie - bezwzgledne")
    option10_3_label.place(x=330, y=720)
    option10_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_10())
    option10_button.place(x=420, y=750)

    option11_0_label = tk.Label(option3_test_window, text="OPCJA NR. 11")
    option11_0_label.place(x=720, y=40)
    option11_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - uniesiona lewa brew")
    option11_1_label.place(x=650, y=65)
    option11_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - uniesiona prawa brew")
    option11_2_label.place(x=650, y=90)
    option11_3_label = tk.Label(option3_test_window, text="Sterowanie - wzgledne")
    option11_3_label.place(x=650, y=115)
    option11_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_11())
    option11_button.place(x=740, y=145)

    option12_0_label = tk.Label(option3_test_window, text="OPCJA NR. 12")
    option12_0_label.place(x=720, y=185)
    option12_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - uniesione brwi")
    option12_1_label.place(x=650, y=210)
    option12_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - otwarte usta")
    option12_2_label.place(x=650, y=235)
    option12_3_label = tk.Label(option3_test_window, text="Sterowanie - wzgledne")
    option12_3_label.place(x=650, y=260)
    option12_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_12())
    option12_button.place(x=740, y=290)

    option13_0_label = tk.Label(option3_test_window, text="OPCJA NR. 13")
    option13_0_label.place(x=720, y=330)
    option13_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - uniesione brwi")
    option13_1_label.place(x=650, y=355)
    option13_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - uśmiech")
    option13_2_label.place(x=650, y=380)
    option13_3_label = tk.Label(option3_test_window, text="Sterowanie - wzgledne")
    option13_3_label.place(x=650, y=405)
    option13_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_13())
    option13_button.place(x=740, y=435)

    option14_0_label = tk.Label(option3_test_window, text="OPCJA NR. 14")
    option14_0_label.place(x=720, y=485)
    option14_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - otwarte usta")
    option14_1_label.place(x=650, y=510)
    option14_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - zamkniete oczy")
    option14_2_label.place(x=650, y=535)
    option14_3_label = tk.Label(option3_test_window, text="Sterowanie - wzgledne")
    option14_3_label.place(x=650, y=560)
    option14_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_14())
    option14_button.place(x=740, y=590)

    option15_0_label = tk.Label(option3_test_window, text="OPCJA NR. 15")
    option15_0_label.place(x=720, y=640)
    option15_1_label = tk.Label(option3_test_window, text="Lewy przycisk myszy - zamkniete lewe oko")
    option15_1_label.place(x=650, y=665)
    option15_2_label = tk.Label(option3_test_window, text="Prawy przycisk myszy - zamkniete prawe oko")
    option15_2_label.place(x=650, y=690)
    option15_3_label = tk.Label(option3_test_window, text="Sterowanie - wzgledne")
    option15_3_label.place(x=650, y=715)
    option15_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_15())
    option15_button.place(x=740, y=750)

    option16_0_label = tk.Label(option3_test_window, text="OPCJA NR. 16")
    option16_0_label.place(x=1070, y=40)
    option16_1_label = tk.Label(option3_test_window, text="Rolka w góre - uniesiona lewa brew")
    option16_1_label.place(x=990, y=65)
    option16_2_label = tk.Label(option3_test_window, text="Rolka w dół - uniesiona prawa brew")
    option16_2_label.place(x=990, y=90)
    option16_3_label = tk.Label(option3_test_window, text="Sterowanie - proste")
    option16_3_label.place(x=990, y=115)
    option16_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_16())
    option16_button.place(x=1100, y=145)

    option17_0_label = tk.Label(option3_test_window, text="OPCJA NR. 17")
    option17_0_label.place(x=1070, y=195)
    option17_1_label = tk.Label(option3_test_window, text="Rolka w góre - uniesione brwi")
    option17_1_label.place(x=990, y=220)
    option17_2_label = tk.Label(option3_test_window, text="Rolka w dół - otwarte usta")
    option17_2_label.place(x=990, y=245)
    option17_3_label = tk.Label(option3_test_window, text="Sterowanie - proste")
    option17_3_label.place(x=990, y=270)
    option17_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_17())
    option17_button.place(x=1100, y=300)

    option3_0_label = tk.Label(option3_test_window, text="OPCJA NR. 18")
    option3_0_label.place(x=1070, y=340)
    option3_1_label = tk.Label(option3_test_window, text="Rolka w góre - uniesione brwi")
    option3_1_label.place(x=990, y=360)
    option3_2_label = tk.Label(option3_test_window, text="Rolka w dół - uśmiech")
    option3_2_label.place(x=990, y=385)
    option3_3_label = tk.Label(option3_test_window, text="Sterowanie - proste")
    option3_3_label.place(x=990, y=410)
    option3_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_18())
    option3_button.place(x=1100, y=440)

    option4_0_label = tk.Label(option3_test_window, text="OPCJA NR. 19")
    option4_0_label.place(x=1070, y=490)
    option4_1_label = tk.Label(option3_test_window, text="Rolka w góre - otwarte usta")
    option4_1_label.place(x=990, y=515)
    option4_2_label = tk.Label(option3_test_window, text="Rolka w dół - zamkniete oczy")
    option4_2_label.place(x=990, y=540)
    option4_3_label = tk.Label(option3_test_window, text="Sterowanie - proste")
    option4_3_label.place(x=990, y=565)
    option4_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_19())
    option4_button.place(x=1100, y=595)

    option5_0_label = tk.Label(option3_test_window, text="OPCJA NR. 20")
    option5_0_label.place(x=1070, y=645)
    option5_1_label = tk.Label(option3_test_window, text="Rolka w góre - zamkniete lewe oko")
    option5_1_label.place(x=990, y=670)
    option5_2_label = tk.Label(option3_test_window, text="Rolka w dół - zamkniete prawe oko")
    option5_2_label.place(x=990, y=695)
    option5_3_label = tk.Label(option3_test_window, text="Sterowanie - proste")
    option5_3_label.place(x=990, y=720)
    option5_button = tk.Button(option3_test_window, text="Start", command=lambda: option3_test_20())
    option5_button.place(x=1100, y=750)
def option3_test_1():
    eyebrow_ar_consec_frames = 3
    counter_eyebrows_left = 0
    total_eyebrows_left = 0
    counter_eyebrows_right = 0
    total_eyebrows_right = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)

        detected_faces = dlib_face_detector(webcam_image_gray, 1)

        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1, (0, 0, 0), 2)
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                A = dist.euclidean(left_eyebrow_landmarks[3], left_eye_landmarks[3])
                B = dist.euclidean(right_eyebrow_landmarks[3], right_eye_landmarks[3])
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if (A > 13 and B < 13):
                        counter_eyebrows_left += 1
                    else:
                        if counter_eyebrows_left >= eyebrow_ar_consec_frames:
                            total_eyebrows_left += 1
                            pyautogui.click(button='left')
                        counter_eyebrows_left = 0

                    if (B > 13 and A < 13):
                        counter_eyebrows_right += 1
                    else:
                        if counter_eyebrows_right >= eyebrow_ar_consec_frames:
                            total_eyebrows_right += 1
                            pyautogui.click(button='right')
                        counter_eyebrows_right = 0

                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

            cv2.putText(webcam_image_gray, "Licznik lewe: {}".format(total_eyebrows_left), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik prawe: {}".format(total_eyebrows_right), (10, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_2():
    eyebrow_ar_thresh = 0.75
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    mouth_ar_thresh = 0.2
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                eyebrow_left_ear = eyebrow_aspect_ratio(left_eyebrow_landmarks, left_eye_landmarks)
                eyebrow_right_ear = eyebrow_aspect_ratio(right_eyebrow_landmarks, right_eye_landmarks)
                ear = (eyebrow_left_ear + eyebrow_right_ear) / 2.0
                mouth_in_ear = eye_aspect_ratio(mouth_in_landmarks)
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1, (0, 0, 0), 2)
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if mouth_in_ear < mouth_ar_thresh:
                        counter_mouth += 1
                    else:
                        if counter_mouth >= mouth_ar_consec_frames:
                            total_mouth += 1
                            pyautogui.click(button='right')
                        counter_mouth = 0


                    if ear < eyebrow_ar_thresh:
                        counter_eyebrows += 1
                    else:
                        if counter_eyebrows >= eyebrow_ar_consec_frames:
                            total_eyebrows += 1
                            pyautogui.click(button='left')
                        counter_eyebrows = 0

                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

            cv2.putText(webcam_image_gray, "Licznik brwi: {}".format(total_eyebrows), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_3():
    eyebrow_ar_thresh = 0.75
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    mouth_ar_thresh = 31
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                eyebrow_left_ear = eyebrow_aspect_ratio(left_eyebrow_landmarks, left_eye_landmarks)
                eyebrow_right_ear = eyebrow_aspect_ratio(right_eyebrow_landmarks, right_eye_landmarks)
                ear = (eyebrow_left_ear + eyebrow_right_ear) / 2.0
                A = dist.euclidean(mouth_out_landmarks[0], mouth_out_landmarks[6])
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if A < mouth_ar_thresh:
                        counter_mouth += 1
                    else:
                        if counter_mouth >= mouth_ar_consec_frames:
                            total_mouth += 1
                            pyautogui.click(button='right')
                        counter_mouth = 0

                    if ear < eyebrow_ar_thresh:
                        counter_eyebrows += 1
                    else:
                        if counter_eyebrows >= eyebrow_ar_consec_frames:
                            total_eyebrows += 1
                            pyautogui.click(button='left')
                        counter_eyebrows = 0

                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

            cv2.putText(webcam_image_gray, "Licznik brwi: {}".format(total_eyebrows), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_4():
    eye_ar_thresh = 0.25
    eye_ar_consec_frames = 3
    counter_eyes = 0
    total_eyes_blinks = 0
    mouth_ar_thresh = 0.2
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
            if len(detected_faces) == 1:
                mouth_in_ear = eye_aspect_ratio(mouth_in_landmarks)
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)
                ear = (left_eye_ear + right_eye_ear) / 2.0
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if ear < eye_ar_thresh:
                        counter_eyes += 1
                    else:
                        if counter_eyes >= eye_ar_consec_frames:
                            total_eyes_blinks += 1
                            pyautogui.click(button='right')
                        counter_eyes = 0

                    if mouth_in_ear < mouth_ar_thresh:
                        counter_mouth += 1
                    else:
                        if counter_mouth >= mouth_ar_consec_frames:
                            total_mouth += 1
                            pyautogui.click(button='left')
                        counter_mouth = 0

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(left_eye_landmarks):
                    left_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eye_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eye_landmarks):
                    right_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eye_position, 1, (0, 0, 0), -1)

                cv2.putText(webcam_image_gray, "Licznik oczy : {}".format(total_eyes_blinks), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_5():
    eye_ar_thresh = 0.3
    eye_ar_consec_frames = 3
    counter_eyes_left = 0
    total_eyes_blinks_left = 0
    counter_eyes_right = 0
    total_eyes_blinks_right = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix(
                [[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
            if len(detected_faces) == 1:
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)

                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if left_eye_ear < eye_ar_thresh - 0.07 and right_eye_ear > eye_ar_thresh - 0.11:
                        counter_eyes_left += 1
                    else:
                        if counter_eyes_left >= eye_ar_consec_frames:
                            total_eyes_blinks_left += 1
                            pyautogui.click(button='left')
                        counter_eyes_left = 0

                    if right_eye_ear < eye_ar_thresh - 0.07 and left_eye_ear > eye_ar_thresh - 0.11:
                        counter_eyes_right += 1
                    else:
                        if counter_eyes_right >= eye_ar_consec_frames:
                            total_eyes_blinks_right += 1
                            pyautogui.click(button='right')
                        counter_eyes_right = 0

                for (i, p) in enumerate(right_eye_landmarks):
                    right_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eye_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(left_eye_landmarks):
                    left_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eye_position, 1, (0, 0, 0), -1)

                cv2.putText(webcam_image_gray, "Licznik oko prawe: {}".format(total_eyes_blinks_right), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(webcam_image_gray, "Licznik oko lewe: {}".format(total_eyes_blinks_left), (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_6():
    eyebrow_ar_consec_frames = 3
    counter_eyebrows_left = 0
    total_eyebrows_left = 0
    counter_eyebrows_right = 0
    total_eyebrows_right = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix(
                [[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                A = dist.euclidean(left_eyebrow_landmarks[3], left_eye_landmarks[3])
                B = dist.euclidean(right_eyebrow_landmarks[3], right_eye_landmarks[3])
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if ((nose_y > 60) and (nose_y < 130)):
                    if ((nose_x > 80) and (nose_x < 160)):
                        if ((nose_x > 130) and (nose_x < 160)):
                            mouse.move(speed, 0)
                        if ((nose_x < 110) and (nose_x > 80)):
                            mouse.move(-speed, 0)
                        if ((nose_y > 60) and (nose_y < 90)):
                            mouse.move(0, -speed)
                        if ((nose_y < 140) and (nose_y > 110)):
                            mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if (A > 13 and B < 13):
                        counter_eyebrows_left += 1
                    else:
                        if counter_eyebrows_left >= eyebrow_ar_consec_frames:
                            total_eyebrows_left += 1
                            pyautogui.click(button='left')
                        counter_eyebrows_left = 0

                    if (B > 13 and A < 13):
                        counter_eyebrows_right += 1
                    else:
                        if counter_eyebrows_right >= eyebrow_ar_consec_frames:
                            total_eyebrows_right += 1
                            pyautogui.click(button='right')
                        counter_eyebrows_right = 0

            for (i, detected_faces) in enumerate(detected_faces):
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y),
                           1, (0, 0, 0), 2)
            cv2.putText(webcam_image_gray, "Licznik lewe: {}".format(total_eyebrows_left), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik prawe: {}".format(total_eyebrows_right), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for (i, p) in enumerate(left_eyebrow_landmarks):
                left_eyebrow_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

            for (i, p) in enumerate(right_eyebrow_landmarks):
                right_eyebrow_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

        cv2.line(webcam_image_gray, (80, 100), (160, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 60), (120, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (80, 60), (160, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_7():
    eyebrow_ar_thresh = 0.75
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    mouth_ar_thresh = 0.2
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix(
                [[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                eyebrow_left_ear = eyebrow_aspect_ratio(left_eyebrow_landmarks, left_eye_landmarks)
                eyebrow_right_ear = eyebrow_aspect_ratio(right_eyebrow_landmarks, right_eye_landmarks)
                ear = (eyebrow_left_ear + eyebrow_right_ear) / 2.0
                mouth_in_ear = eye_aspect_ratio(mouth_in_landmarks)
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y
                if ((nose_y > 60) and (nose_y < 130)):
                    if ((nose_x > 80) and (nose_x < 160)):
                        if ((nose_x > 130) and (nose_x < 160)):
                            mouse.move(speed, 0)
                        if ((nose_x < 110) and (nose_x > 80)):
                            mouse.move(-speed, 0)
                        if ((nose_y > 60) and (nose_y < 90)):
                            mouse.move(0, -speed)
                        if ((nose_y < 140) and (nose_y > 110)):
                            mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if mouth_in_ear < mouth_ar_thresh:
                        counter_mouth += 1
                    else:
                        if counter_mouth >= mouth_ar_consec_frames:
                            total_mouth += 1
                            pyautogui.click(button='right')
                        counter_mouth = 0

                    if ear < eyebrow_ar_thresh:
                        counter_eyebrows += 1
                    else:
                        if counter_eyebrows >= eyebrow_ar_consec_frames:
                            total_eyebrows += 1
                            pyautogui.click(button='left')
                        counter_eyebrows = 0

                for (i, detected_faces) in enumerate(detected_faces):
                    cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y),
                               1, (0, 0, 0), 2)
                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

            cv2.putText(webcam_image_gray, "Licznik brwi: {}".format(total_eyebrows), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (80, 100), (160, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 60), (120, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (80, 60), (160, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_8():
    eyebrow_ar_thresh = 0.75
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    mouth_ar_thresh = 31
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                eyebrow_left_ear = eyebrow_aspect_ratio(left_eyebrow_landmarks, left_eye_landmarks)
                eyebrow_right_ear = eyebrow_aspect_ratio(right_eyebrow_landmarks, right_eye_landmarks)
                ear = (eyebrow_left_ear + eyebrow_right_ear) / 2.0
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                A = dist.euclidean(mouth_out_landmarks[0], mouth_out_landmarks[6])
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if ((nose_y > 60) and (nose_y < 130)):
                    if ((nose_x > 80) and (nose_x < 160)):
                        if ((nose_x > 130) and (nose_x < 160)):
                            mouse.move(speed, 0)
                        if ((nose_x < 110) and (nose_x > 80)):
                            mouse.move(-speed, 0)
                        if ((nose_y > 60) and (nose_y < 90)):
                            mouse.move(0, -speed)
                        if ((nose_y < 140) and (nose_y > 110)):
                            mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if A < mouth_ar_thresh:
                        counter_mouth += 1
                    else:
                        if counter_mouth >= mouth_ar_consec_frames:
                            total_mouth += 1
                            pyautogui.click(button='right')
                        counter_mouth = 0

                    if ear < eyebrow_ar_thresh:
                        counter_eyebrows += 1
                    else:
                        if counter_eyebrows >= eyebrow_ar_consec_frames:
                            total_eyebrows += 1
                            pyautogui.click(button='left')
                        counter_eyebrows = 0

                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

            cv2.putText(webcam_image_gray, "Licznik brwi: {}".format(total_eyebrows), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (80, 100), (160, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 60), (120, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (80, 60), (160, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_9():
    eye_ar_thresh = 0.25
    eye_ar_consec_frames = 3
    counter_eyes = 0
    total_eyes_blinks = 0
    mouth_ar_thresh = 0.2
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
            if len(detected_faces) == 1:
                mouth_in_ear = eye_aspect_ratio(mouth_in_landmarks)
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)
                ear = (left_eye_ear + right_eye_ear) / 2.0
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1, (0, 0, 0), 2)
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if ((nose_y > 60) and (nose_y < 130)):
                    if ((nose_x > 80) and (nose_x < 160)):
                        if ((nose_x > 130) and (nose_x < 160)):
                            mouse.move(speed, 0)
                        if ((nose_x < 110) and (nose_x > 80)):
                            mouse.move(-speed, 0)
                        if ((nose_y > 60) and (nose_y < 90)):
                            mouse.move(0, -speed)
                        if ((nose_y < 140) and (nose_y > 110)):
                            mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if ear < eye_ar_thresh:
                        counter_eyes += 1
                    else:
                        if counter_eyes >= eye_ar_consec_frames:
                            total_eyes_blinks += 1
                            pyautogui.click(button='right')
                        counter_eyes = 0

                    if mouth_in_ear < mouth_ar_thresh:
                        counter_mouth += 1
                    else:
                        if counter_mouth >= mouth_ar_consec_frames:
                            total_mouth += 1
                            pyautogui.click(button='left')
                        counter_mouth = 0

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(left_eye_landmarks):
                    left_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eye_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eye_landmarks):
                    right_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eye_position, 1, (0, 0, 0), -1)

                cv2.putText(webcam_image_gray, "Licznik oczy : {}".format(total_eyes_blinks), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (80, 100), (160, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 60), (120, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (80, 60), (160, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_10():
    eye_ar_thresh = 0.3
    eye_ar_consec_frames = 3
    counter_eyes_left = 0
    total_eyes_blinks_left = 0
    counter_eyes_right = 0
    total_eyes_blinks_right = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix(
                [[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]

            if len(detected_faces) == 1:
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1, (0, 0, 0), 2)
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if ((nose_y > 60) and (nose_y < 130)):
                    if ((nose_x > 80) and (nose_x < 160)):
                        if ((nose_x > 130) and (nose_x < 160)):
                            mouse.move(speed, 0)
                        if ((nose_x < 110) and (nose_x > 80)):
                            mouse.move(-speed, 0)
                        if ((nose_y > 60) and (nose_y < 90)):
                            mouse.move(0, -speed)
                        if ((nose_y < 140) and (nose_y > 110)):
                            mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if left_eye_ear < eye_ar_thresh - 0.07 and right_eye_ear > eye_ar_thresh - 0.11:
                        counter_eyes_left += 1
                    else:
                        if counter_eyes_left >= eye_ar_consec_frames:
                            total_eyes_blinks_left += 1
                            pyautogui.click(button='left')
                        counter_eyes_left = 0

                    if right_eye_ear < eye_ar_thresh - 0.07 and left_eye_ear > eye_ar_thresh - 0.11:
                        counter_eyes_right += 1
                    else:
                        if counter_eyes_right >= eye_ar_consec_frames:
                            total_eyes_blinks_right += 1
                            pyautogui.click(button='right')
                        counter_eyes_right = 0

                for (i, p) in enumerate(right_eye_landmarks):
                    right_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eye_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(left_eye_landmarks):
                    left_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eye_position, 1, (0, 0, 0), -1)

                cv2.putText(webcam_image_gray, "Licznik oko prawe: {}".format(total_eyes_blinks_right), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(webcam_image_gray, "Licznik oko lewe: {}".format(total_eyes_blinks_left), (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (80, 100), (160, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 60), (120, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (80, 60), (160, 140), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_11():
    eyebrow_ar_consec_frames = 3
    counter_eyebrows_left = 0
    total_eyebrows_left = 0
    counter_eyebrows_right = 0
    total_eyebrows_right = 0
    sx = GetSystemMetrics(0)
    sy = GetSystemMetrics(1)
    (camx, camy) = (320, 240)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix(
                [[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                A = dist.euclidean(left_eyebrow_landmarks[3], left_eye_landmarks[3])
                B = dist.euclidean(right_eyebrow_landmarks[3], right_eye_landmarks[3])

                if (A > 13 and B < 13):
                    counter_eyebrows_left += 1
                else:
                    if counter_eyebrows_left >= eyebrow_ar_consec_frames:
                        total_eyebrows_left += 1
                        pyautogui.click(button='left')
                    counter_eyebrows_left = 0

                if (B > 13 and A < 13):
                    counter_eyebrows_right += 1
                else:
                    if counter_eyebrows_right >= eyebrow_ar_consec_frames:
                        total_eyebrows_right += 1
                        pyautogui.click(button='right')
                    counter_eyebrows_right = 0

                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y
                pyautogui.moveTo(sx - ((nose_x * sx / camx) * 2), ((nose_y * sy / camy)))

            cv2.putText(webcam_image_gray, "Licznik lewe: {}".format(total_eyebrows_left), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik prawe: {}".format(total_eyebrows_right), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for (i, p) in enumerate(left_eyebrow_landmarks):
                left_eyebrow_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

            for (i, p) in enumerate(right_eyebrow_landmarks):
                right_eyebrow_position = (p[0, 0], p[0, 1])
                cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_12():
    eyebrow_ar_thresh = 0.75
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    mouth_ar_thresh = 0.2
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sx = GetSystemMetrics(0)
    sy = GetSystemMetrics(1)
    (camx, camy) = (320, 240)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix(
                [[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                eyebrow_left_ear = eyebrow_aspect_ratio(left_eyebrow_landmarks, left_eye_landmarks)
                eyebrow_right_ear = eyebrow_aspect_ratio(right_eyebrow_landmarks, right_eye_landmarks)
                ear = (eyebrow_left_ear + eyebrow_right_ear) / 2.0
                mouth_in_ear = eye_aspect_ratio(mouth_in_landmarks)

                if mouth_in_ear < mouth_ar_thresh:
                    counter_mouth += 1
                else:
                    if counter_mouth >= mouth_ar_consec_frames:
                        total_mouth += 1
                        pyautogui.click(button='right')
                    counter_mouth = 0

                if ear < eyebrow_ar_thresh:
                    counter_eyebrows += 1
                else:
                    if counter_eyebrows >= eyebrow_ar_consec_frames:
                        total_eyebrows += 1
                        pyautogui.click(button='left')
                    counter_eyebrows = 0

                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y
                pyautogui.moveTo(sx - ((nose_x * sx / camx) * 2), ((nose_y * sy / camy)))

                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

            cv2.putText(webcam_image_gray, "Licznik brwi: {}".format(total_eyebrows), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_13():
    eyebrow_ar_thresh = 0.75
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    mouth_ar_thresh = 31
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sx = GetSystemMetrics(0)
    sy = GetSystemMetrics(1)
    (camx, camy) = (320, 240)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                eyebrow_left_ear = eyebrow_aspect_ratio(left_eyebrow_landmarks, left_eye_landmarks)
                eyebrow_right_ear = eyebrow_aspect_ratio(right_eyebrow_landmarks, right_eye_landmarks)
                ear = (eyebrow_left_ear + eyebrow_right_ear) / 2.0
                A = dist.euclidean(mouth_out_landmarks[0], mouth_out_landmarks[6])
                if A < mouth_ar_thresh:
                    counter_mouth += 1
                else:
                    if counter_mouth >= mouth_ar_consec_frames:
                        total_mouth += 1
                        pyautogui.click(button='right')
                    counter_mouth = 0

                if ear < eyebrow_ar_thresh:
                    counter_eyebrows += 1
                else:
                    if counter_eyebrows >= eyebrow_ar_consec_frames:
                        total_eyebrows += 1
                        pyautogui.click(button='left')
                    counter_eyebrows = 0

                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y
                pyautogui.moveTo(sx - ((nose_x * sx / camx) * 2), ((nose_y * sy / camy)))

                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

            cv2.putText(webcam_image_gray, "Licznik brwi: {}".format(total_eyebrows), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_14():
    eye_ar_thresh = 0.25
    eye_ar_consec_frames = 3
    counter_eyes = 0
    total_eyes_blinks = 0
    mouth_ar_thresh = 0.2
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sx = GetSystemMetrics(0)
    sy = GetSystemMetrics(1)
    (camx, camy) = (320, 240)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
            if len(detected_faces) == 1:
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                mouth_in_ear = eye_aspect_ratio(mouth_in_landmarks)
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)
                ear = (left_eye_ear + right_eye_ear) / 2.0

                if ear < eye_ar_thresh:
                    counter_eyes += 1
                else:
                    if counter_eyes >= eye_ar_consec_frames:
                        total_eyes_blinks += 1
                        pyautogui.click(button='right')
                    counter_eyes = 0

                if mouth_in_ear < mouth_ar_thresh:
                    counter_mouth += 1
                else:
                    if counter_mouth >= mouth_ar_consec_frames:
                        total_mouth += 1
                        pyautogui.click(button='left')
                    counter_mouth = 0

                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y
                pyautogui.moveTo(sx - ((nose_x * sx / camx) * 2), ((nose_y * sy / camy)))

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(left_eye_landmarks):
                    left_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eye_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eye_landmarks):
                    right_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eye_position, 1, (0, 0, 0), -1)

                cv2.putText(webcam_image_gray, "Licznik oczy : {}".format(total_eyes_blinks), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_15():
    eye_ar_thresh = 0.3
    eye_ar_consec_frames = 3
    counter_eyes_left = 0
    total_eyes_blinks_left = 0
    counter_eyes_right = 0
    total_eyes_blinks_right = 0
    sx = GetSystemMetrics(0)
    sy = GetSystemMetrics(1)
    (camx, camy) = (320, 240)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix(
                [[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
            if len(detected_faces) == 1:
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                if left_eye_ear < eye_ar_thresh - 0.07 and right_eye_ear > eye_ar_thresh - 0.11:
                    counter_eyes_left += 1
                else:
                    if counter_eyes_left >= eye_ar_consec_frames:
                        total_eyes_blinks_left += 1
                        pyautogui.click(button='left')
                    counter_eyes_left = 0

                if right_eye_ear < eye_ar_thresh - 0.07 and left_eye_ear > eye_ar_thresh - 0.11:
                    counter_eyes_right += 1
                else:
                    if counter_eyes_right >= eye_ar_consec_frames:
                        total_eyes_blinks_right += 1
                        pyautogui.click(button='right')
                    counter_eyes_right = 0

                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y
                pyautogui.moveTo(sx - ((nose_x * sx / camx) * 2), ((nose_y * sy / camy)))

                for (i, p) in enumerate(right_eye_landmarks):
                    right_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eye_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(left_eye_landmarks):
                    left_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eye_position, 1, (0, 0, 0), -1)

                cv2.putText(webcam_image_gray, "Licznik oko prawe: {}".format(total_eyes_blinks_right), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(webcam_image_gray, "Licznik oko lewe: {}".format(total_eyes_blinks_left), (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_16():
    eyebrow_ar_consec_frames = 3
    counter_eyebrows_left = 0
    total_eyebrows_left = 0
    counter_eyebrows_right = 0
    total_eyebrows_right = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)

        detected_faces = dlib_face_detector(webcam_image_gray, 1)

        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1, (0, 0, 0), 2)
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                A = dist.euclidean(left_eyebrow_landmarks[3], left_eye_landmarks[3])
                B = dist.euclidean(right_eyebrow_landmarks[3], right_eye_landmarks[3])
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if (A > 13 and B < 13):
                        counter_eyebrows_left += 1
                    else:
                        if counter_eyebrows_left >= eyebrow_ar_consec_frames:
                            total_eyebrows_left += 1
                        counter_eyebrows_left = 0
                        pyautogui.scroll(15)

                    if (B > 13 and A < 13):
                        counter_eyebrows_right += 1
                    else:
                        if counter_eyebrows_right >= eyebrow_ar_consec_frames:
                            total_eyebrows_right += 1
                        counter_eyebrows_right = 0
                        pyautogui.scroll(-15)

                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

            cv2.putText(webcam_image_gray, "Licznik lewe: {}".format(total_eyebrows_left), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik prawe: {}".format(total_eyebrows_right), (10, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_17():
    eyebrow_ar_thresh = 0.75
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    mouth_ar_thresh = 0.2
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix(
                [[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                eyebrow_left_ear = eyebrow_aspect_ratio(left_eyebrow_landmarks, left_eye_landmarks)
                eyebrow_right_ear = eyebrow_aspect_ratio(right_eyebrow_landmarks, right_eye_landmarks)
                ear = (eyebrow_left_ear + eyebrow_right_ear) / 2.0
                mouth_in_ear = eye_aspect_ratio(mouth_in_landmarks)
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if mouth_in_ear < mouth_ar_thresh:
                        counter_mouth += 1
                    else:
                        if counter_mouth >= mouth_ar_consec_frames:
                            total_mouth += 1
                            counter_mouth = 0
                        pyautogui.scroll(-15)
                    if ear < eyebrow_ar_thresh:
                        counter_eyebrows += 1
                    else:
                        if counter_eyebrows >= eyebrow_ar_consec_frames:
                            total_eyebrows += 1
                        counter_eyebrows = 0
                        pyautogui.scroll(15)

                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

            cv2.putText(webcam_image_gray, "Licznik brwi: {}".format(total_eyebrows), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_18():
    eyebrow_ar_thresh = 0.75
    eyebrow_ar_consec_frames = 3
    counter_eyebrows = 0
    total_eyebrows = 0
    mouth_ar_thresh = 31
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eyebrow_landmarks = dlib_landmarks[dlib_left_eyebrow_landmarks]
            right_eyebrow_landmarks = dlib_landmarks[dlib_right_eyebrow_landmarks]

            if len(detected_faces) == 1:
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
                right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
                eyebrow_left_ear = eyebrow_aspect_ratio(left_eyebrow_landmarks, left_eye_landmarks)
                eyebrow_right_ear = eyebrow_aspect_ratio(right_eyebrow_landmarks, right_eye_landmarks)
                ear = (eyebrow_left_ear + eyebrow_right_ear) / 2.0
                A = dist.euclidean(mouth_out_landmarks[0], mouth_out_landmarks[6])
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if A < mouth_ar_thresh:
                        counter_mouth += 1
                    else:
                        if counter_mouth >= mouth_ar_consec_frames:
                            total_mouth += 1
                        counter_mouth = 0
                        pyautogui.scroll(-15)

                    if ear < eyebrow_ar_thresh:
                        counter_eyebrows += 1
                    else:
                        if counter_eyebrows >= eyebrow_ar_consec_frames:
                            total_eyebrows += 1
                        counter_eyebrows = 0
                        pyautogui.scroll(15)

                for (i, p) in enumerate(left_eyebrow_landmarks):
                    left_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eyebrow_landmarks):
                    right_eyebrow_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eyebrow_position, 2, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

            cv2.putText(webcam_image_gray, "Licznik brwi: {}".format(total_eyebrows), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_19():
    eye_ar_thresh = 0.25
    eye_ar_consec_frames = 3
    counter_eyes = 0
    total_eyes_blinks = 0
    mouth_ar_thresh = 0.2
    mouth_ar_consec_frames = 3
    counter_mouth = 0
    total_mouth = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix([[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            mouth_out_landmarks = dlib_landmarks[dlib_mouth_out_landmarks]
            mouth_in_landmarks = dlib_landmarks[dlib_mouth_in_landmarks]
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
            if len(detected_faces) == 1:
                mouth_in_ear = eye_aspect_ratio(mouth_in_landmarks)
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)
                ear = (left_eye_ear + right_eye_ear) / 2.0
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)
                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if ear < eye_ar_thresh:
                        counter_eyes += 1
                    else:
                        if counter_eyes >= eye_ar_consec_frames:
                            total_eyes_blinks += 1
                        counter_eyes = 0
                        pyautogui.scroll(-15)

                    if mouth_in_ear < mouth_ar_thresh:
                        counter_mouth += 1
                    else:
                        if counter_mouth >= mouth_ar_consec_frames:
                            total_mouth += 1
                        counter_mouth = 0
                        pyautogui.scroll(15)

                for (i, p) in enumerate(mouth_out_landmarks):
                    mouth_out_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_out_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(mouth_in_landmarks):
                    mouth_in_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, mouth_in_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(left_eye_landmarks):
                    left_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eye_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(right_eye_landmarks):
                    right_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eye_position, 1, (0, 0, 0), -1)

                cv2.putText(webcam_image_gray, "Licznik oczy : {}".format(total_eyes_blinks), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(webcam_image_gray, "Licznik usta: {}".format(total_mouth), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()
def option3_test_20():
    eye_ar_thresh = 0.3
    eye_ar_consec_frames = 3
    counter_eyes_left = 0
    total_eyes_blinks_left = 0
    counter_eyes_right = 0
    total_eyes_blinks_right = 0
    sensitivity_mouse = sensitivity.get()
    speed = int(sensitivity_mouse)
    webcam_capture = cv2.VideoCapture(0)
    while webcam_capture.isOpened():
        webcam_return, webcam_image_color = webcam_capture.read()
        webcam_image_gray = cv2.cvtColor(webcam_image_color, cv2.COLOR_BGR2GRAY)
        webcam_image_gray = imutils.resize(webcam_image_gray, height=320, width=240)
        webcam_image_gray = cv2.flip(webcam_image_gray, 1)
        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        detected_faces = dlib_face_detector(webcam_image_gray, 1)
        cv2.putText(webcam_image_gray, "Wykryte osoby: {}".format(len(detected_faces)), (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, detected_face) in enumerate(detected_faces):
            dlib_landmarks = np.matrix(
                [[p.x, p.y] for p in dlib_facial_landmark_predictor(webcam_image_gray, detected_face).parts()])
            left_eye_landmarks = dlib_landmarks[dlib_left_eye_landmarks]
            right_eye_landmarks = dlib_landmarks[dlib_right_eye_landmarks]
            if len(detected_faces) == 1:
                left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ear = eye_aspect_ratio(right_eye_landmarks)
                landmarks_coordinates = dlib_facial_landmark_predictor(webcam_image_gray, detected_face)
                cv2.circle(webcam_image_gray, (landmarks_coordinates.part(30).x, landmarks_coordinates.part(30).y), 1,
                           (0, 0, 0), 2)

                nose_x = landmarks_coordinates.part(30).x
                nose_y = landmarks_coordinates.part(30).y

                if (nose_x > 130):
                    mouse.move(speed, 0)
                if (nose_x < 110):
                    mouse.move(-speed, 0)
                if (nose_y < 90):
                    mouse.move(0, -speed)
                if (nose_y > 110):
                    mouse.move(0, speed)

                if (nose_x < 130) and (nose_x > 110) and (nose_y < 110) and (nose_y > 90):
                    if left_eye_ear < eye_ar_thresh - 0.07 and right_eye_ear > eye_ar_thresh - 0.11:
                        counter_eyes_left += 1
                    else:
                        if counter_eyes_left >= eye_ar_consec_frames:
                            total_eyes_blinks_left += 1
                        counter_eyes_left = 0
                        pyautogui.scroll(15)

                    if right_eye_ear < eye_ar_thresh - 0.07 and left_eye_ear > eye_ar_thresh - 0.11:
                        counter_eyes_right += 1
                    else:
                        if counter_eyes_right >= eye_ar_consec_frames:
                            total_eyes_blinks_right += 1
                        counter_eyes_right = 0
                        pyautogui.scroll(-15)

                for (i, p) in enumerate(right_eye_landmarks):
                    right_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, right_eye_position, 1, (0, 0, 0), -1)

                for (i, p) in enumerate(left_eye_landmarks):
                    left_eye_position = (p[0, 0], p[0, 1])
                    cv2.circle(webcam_image_gray, left_eye_position, 1, (0, 0, 0), -1)

                cv2.putText(webcam_image_gray, "Licznik oko prawe: {}".format(total_eyes_blinks_right), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(webcam_image_gray, "Licznik oko lewe: {}".format(total_eyes_blinks_left), (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.line(webcam_image_gray, (0, 100), (320, 100), (0, 0, 0), 1)
        cv2.line(webcam_image_gray, (120, 0), (120, 240), (0, 0, 0), 1)
        cv2.rectangle(webcam_image_gray, (110, 90), (130, 110), (0, 0, 0), 1)

        cv2.imshow('Webcam capture - gray', webcam_image_gray)
        if cv2.waitKey(1) == 27:
            break
    webcam_capture.release()
    cv2.destroyAllWindows()


tk.mainloop()