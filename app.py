import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter
import pyttsx3

st.title("Sign Language Detector")

# Load the trained model
model_dict = pickle.load(open("model.p", "rb"))
model = model_dict["model"]

# Configuration
CONFIRM_FRAMES = 20
CONFIDENCE_THRESHOLD = 0.80

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# Text-to-speech engine
engine = pyttsx3.init()

# Session state for Streamlit
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "prediction_buffer" not in st.session_state:
    st.session_state.prediction_buffer = deque(maxlen=CONFIRM_FRAMES)
if "gesture_locked" not in st.session_state:
    st.session_state.gesture_locked = False
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "cap" not in st.session_state:
    st.session_state.cap = None

# Buttons to control camera
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Start Camera", key="start_cam"):
        st.session_state.run_camera = True
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)
with col2:
    if st.button("Stop Camera", key="stop_cam"):
        st.session_state.run_camera = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
with col3:
    if st.button("Reset Sentence", key="reset_sent"):
        st.session_state.sentence = ""

# Speak sentence button
if st.button("Speak Sentence", key="speak_btn") and st.session_state.sentence.strip():
    engine.say(st.session_state.sentence)
    engine.runAndWait()

# Frame display
FRAME_WINDOW = st.image([], channels="BGR", caption="Camera Feed", use_column_width=True)

# Main camera loop
if st.session_state.run_camera:
    cap = st.session_state.cap
    if cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        cap = st.session_state.cap

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_list, y_list = [], []
                for lm in hand_landmarks.landmark:
                    x_list.append(lm.x)
                    y_list.append(lm.y)
                data = []
                for lm in hand_landmarks.landmark:
                    data.append(lm.x - min(x_list))
                    data.append(lm.y - min(y_list))

                probabilities = model.predict_proba([np.asarray(data)])[0]
                max_prob = np.max(probabilities)
                predicted = model.classes_[np.argmax(probabilities)]

                if max_prob > CONFIDENCE_THRESHOLD:
                    st.session_state.prediction_buffer.append(predicted)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            st.session_state.gesture_locked = False
            st.session_state.prediction_buffer.clear()

        if len(st.session_state.prediction_buffer) == CONFIRM_FRAMES and not st.session_state.gesture_locked:
            final_letter = Counter(st.session_state.prediction_buffer).most_common(1)[0][0]
            if final_letter == "SPACE":
                st.session_state.sentence += " "
            elif final_letter == "DELETE":
                st.session_state.sentence = st.session_state.sentence[:-1]
            else:
                st.session_state.sentence += final_letter
            st.session_state.gesture_locked = True
            st.session_state.prediction_buffer.clear()

        FRAME_WINDOW.image(frame, channels="BGR")

# Display current sentence
st.text_area("Detected Sentence", value=st.session_state.sentence, height=100, key="sentence_area")