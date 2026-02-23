import cv2
import mediapipe as mp
import csv
import os

gesture = input("Enter gesture (A-Z, SPACE, DELETE): ").upper()
num_samples = 300

data_dir = os.path.join("Data", "alphabets", gesture)
os.makedirs(data_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

count = 0
print("Collecting normalized data for", gesture)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

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

            if count < num_samples:
                file_path = os.path.join(data_dir, f"{gesture}_{count}.csv")
                with open(file_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                count += 1

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"{gesture}: {count}/{num_samples}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_samples:
        break

cap.release()
cv2.destroyAllWindows()
print("Done collecting", gesture)