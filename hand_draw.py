import cv2
import mediapipe as mp
import numpy as np
import math

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

canvas = None
prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            ix = int(hand_landmarks.landmark[8].x * w)
            iy = int(hand_landmarks.landmark[8].y * h)

            mx = int(hand_landmarks.landmark[12].x * w)
            my = int(hand_landmarks.landmark[12].y * h)

            rx = int(hand_landmarks.landmark[16].x * w)
            ry = int(hand_landmarks.landmark[16].y * h)

            d1 = math.hypot(ix - mx, iy - my)
            d2 = math.hypot(mx - rx, my - ry)

            if d1 < 40 and d2 < 40:
                canvas = np.zeros_like(frame)
                prev_x, prev_y = None, None
            else:
                if prev_x is None:
                    prev_x, prev_y = ix, iy

                cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0, 255, 0), 5)
                prev_x, prev_y = ix, iy

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = None, None

    combined = cv2.add(frame, canvas)
    cv2.imshow("El Hareketi ile Cizim", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()