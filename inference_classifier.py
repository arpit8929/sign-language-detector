import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)

# Labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 27: 'Done', 28: 'Thank You',
    29: 'I Love You', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.', 33: 'SPACE'
}

# Storage for detected text
recognized_text = ""
prev_time = time.time()
char_delay = 0.5
word_delay = 1.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            x_, y_, data_aux = [], [], []
            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            # Show hand coordinates (for debugging)
            for idx, (x, y) in enumerate(zip(x_, y_)):
                cx, cy = int(x * W), int(y * H)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(idx), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Bounding box
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Ensure correct feature count for model
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Handling text formation
                current_time = time.time()
                time_diff = current_time - prev_time

                if predicted_character == 'SPACE':
                    if recognized_text and recognized_text[-1] != ' ':
                        recognized_text += ' '
                else:
                    if time_diff > word_delay:
                        recognized_text += ' ' + predicted_character
                    elif time_diff > char_delay:
                        recognized_text += predicted_character

                prev_time = current_time

                # Display prediction
                cv2.putText(frame, predicted_character, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Display recognized text
    cv2.putText(frame, "Text: " + recognized_text, (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
