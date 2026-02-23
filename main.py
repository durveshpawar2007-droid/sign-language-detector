import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
from collections import deque, Counter

CONFIRM_FRAMES = 20
CONFIDENCE_THRESHOLD = 0.80

model_dict = pickle.load(open("model.p", "rb"))
model = model_dict["model"]

engine = pyttsx3.init()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,              # Better landmark precision
    min_detection_confidence=0.7,    # Slightly relaxed
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Improve camera quality (this helps finger detection A LOT)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

sentence = ""
prediction_buffer = deque(maxlen=CONFIRM_FRAMES)
gesture_locked = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # Improve brightness slightly
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_prediction = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            x_list = []
            y_list = []

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
                current_prediction = predicted
                prediction_buffer.append(predicted)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        gesture_locked = False
        prediction_buffer.clear()

    if len(prediction_buffer) == CONFIRM_FRAMES and not gesture_locked:
        final_letter = Counter(prediction_buffer).most_common(1)[0][0]

        if final_letter == "SPACE":
            sentence += " "
        elif final_letter == "DELETE":
            sentence = sentence[:-1]
        else:
            sentence += final_letter

        gesture_locked = True
        prediction_buffer.clear()

    cv2.putText(frame, f"Sentence: {sentence}",
                (20, H - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.imshow("Alphabet Detector", frame)

    key = cv2.waitKey(1)

    if key == 13:
        if sentence.strip():
            engine.say(sentence)
            engine.runAndWait()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()