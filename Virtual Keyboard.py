import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand and Drawing Utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Desired display size
display_width = 1280
display_height = 720

# Initialize video capture
video = cv2.VideoCapture(0)

# Set the resolution of the video capture
video.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

# Define the keyboard layout
keyboard_keys = [
    "QWERTYUIOP",
    "ASDFGHJKL",
    "ZXCVBNM"
]
key_size = 100
key_padding = 10

# Calculate keyboard starting points
keyboard_start_x = (display_width - (key_size * len(keyboard_keys[0]) + key_padding * (len(keyboard_keys[0]) - 1))) // 2
keyboard_start_y = 50  # Place the keyboard at the top part of the screen

# Initialize word
word = ""

# Function to draw the keyboard on the frame
def draw_keyboard(frame):
    for i, row in enumerate(keyboard_keys):
        for j, key in enumerate(row):
            x = keyboard_start_x + j * (key_size + key_padding)
            y = keyboard_start_y + i * (key_size + key_padding)
            cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), (255, 255, 255), 2)
            cv2.putText(frame, key, (x + 20, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

# Function to detect which key is being pointed at
def detect_key(x, y):
    for i, row in enumerate(keyboard_keys):
        for j, key in enumerate(row):
            key_x = keyboard_start_x + j * (key_size + key_padding)
            key_y = keyboard_start_y + i * (key_size + key_padding)
            if key_x < x < key_x + key_size and key_y < y < key_y + key_size:
                return key
    return None

# Initialize the Hand Tracker
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB before processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hand landmarks
        results = hands.process(image_rgb)

        # Convert the image color back so it can be displayed
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw the virtual keyboard
        draw_keyboard(image_bgr)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    image_bgr, hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=5),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # Get index finger tip landmark
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_pixel = mp_drawing._normalized_to_pixel_coordinates(
                    index_finger_tip.x, index_finger_tip.y, display_width, display_height
                )

                if index_finger_pixel:
                    current_x, current_y = index_finger_pixel

                    # Detect which key is being pointed at
                    key = detect_key(current_x, current_y)
                    if key:
                        cv2.putText(image_bgr, f"Clicked: {key}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (0, 255, 0), 3)
                        if not word.endswith(key):
                            word += key

        # Display the typed word
        cv2.putText(image_bgr, f"Word: {word}", (10, display_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 3)

        # Display the output with virtual keyboard
        cv2.imshow("Virtual Keyboard with Hand Gestures", image_bgr)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release video capture object and close display windows
video.release()
cv2.destroyAllWindows()
