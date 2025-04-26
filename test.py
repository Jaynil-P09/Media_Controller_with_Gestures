#Control media using gestures
import cv2
import mediapipe as mp
import pyautogui
import time

# Mediapipe setup with optimized config
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.6,
    model_complexity=0,
    static_image_mode=False
)
mp_draw = mp.solutions.drawing_utils

gesture_cooldown = 1.0
last_gesture_time = 0
playpause_triggered = False

def count_extended_fingers(landmarks):
    fingers = []

    if landmarks[4].y < landmarks[2].y:
        thumb_extended = 1
    else:
        thumb_extended = 0

    finger_tips = [8, 12, 16, 20]
    for tip in finger_tips:
        fingers.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)

    return fingers, thumb_extended

def get_hand_sign(landmarks, label):
    fingers, thumb_extended = count_extended_fingers(landmarks)
    total_fingers = sum(fingers)

    thumb_tip = landmarks[4]
    thumb_base = landmarks[2]
    thumb_ip = landmarks[3]

    thumb_up = (thumb_tip.y < thumb_base.y) and (thumb_ip.y < thumb_base.y)
    thumb_down = (thumb_tip.y > thumb_base.y) and (thumb_ip.y > thumb_base.y)

    if total_fingers == 4:
        return "playpause"
    elif total_fingers == 1:
        return "next"
    elif total_fingers == 2:
        return "previous"
    elif total_fingers == 0:
        if label == "Right":
            if thumb_down and (thumb_tip.y > thumb_base.y + 0.02):
                return "volumedown"
            elif thumb_up:
                return "volumeup"
    return None

# Webcam start
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and flip frame for speed
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for handLms, handLabel in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handLabel.classification[0].label  # 'Left' or 'Right'
            lm_list = handLms.landmark

            # Only draw if needed (commented for speed)
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            sign = get_hand_sign(lm_list, label)
            current_time = time.time()

            if sign:
                if (current_time - last_gesture_time) > gesture_cooldown:
                    if label == "Right":
                        if sign == "volumeup":
                            pyautogui.press("volumeup")
                            print("volumeup")
                        elif sign == "volumedown":
                            pyautogui.press("volumedown")
                            print("volumedown")
                    else:
                        if sign == "playpause" and not playpause_triggered:
                            pyautogui.press("playpause")
                            print("play/pause")
                            playpause_triggered = True
                        elif sign == "next":
                            pyautogui.press("nexttrack")
                            print("nexttrack")
                        elif sign == "previous":
                            pyautogui.press("prevtrack")
                            print("prevtrack")

                    last_gesture_time = current_time
            else:
                playpause_triggered = False

    cv2.imshow("Hand Sign Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()












