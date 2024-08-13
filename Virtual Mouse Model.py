import mediapipe as mp
import cv2
import pyautogui
import win32api

# Initialize Mediapipe and OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize video capture
video = cv2.VideoCapture(0)

# Screen resolution
screen_width, screen_height = pyautogui.size()

# Mouse smoothing parameters
SMOOTHING_FACTOR = 0.3
previous_x, previous_y = 0, 0

# Initialize the Hand Tracker
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB before processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get the dimensions of the image
        image_height, image_width, _ = image.shape

        # Process the image and find hand landmarks
        results = hands.process(image)

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks and connections
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw a circle around the index finger tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pixel_coordinates = mp_drawing._normalized_to_pixel_coordinates(
                    index_finger_tip.x, index_finger_tip.y, image_width, image_height
                )
                if pixel_coordinates:
                    cv2.circle(image, pixel_coordinates, 10, (0, 255, 0), 2)

                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )

                # Check finger states
                finger_states = []
                for landmark in mp_hands.HandLandmark:
                    if landmark in [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]:
                        finger_landmark = hand_landmarks.landmark[landmark]
                        # Simple threshold-based approach for finger states
                        if landmark in [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]:
                            distance = abs(finger_landmark.y - hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)
                            finger_states.append(distance < 0.1)  # Adjust threshold as needed
                        else:
                            finger_states.append(finger_landmark.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)

                # Debugging outputs
                print("Finger States:", finger_states)

                # Perform action based on finger states
                if all(finger_states[1:]) and not finger_states[0]:  # All fingers except index are closed
                    # Move cursor based on index finger position
                    if pixel_coordinates:
                        current_x = pixel_coordinates[0] * screen_width / image_width
                        current_y = pixel_coordinates[1] * screen_height / image_height

                        # Debugging outputs
                        print("Cursor Coordinates:", current_x, current_y)

                        # Apply smoothing
                        smoothed_x = previous_x + (current_x - previous_x) * SMOOTHING_FACTOR
                        smoothed_y = previous_y + (current_y - previous_y) * SMOOTHING_FACTOR

                        # Ensure coordinates are within screen bounds
                        smoothed_x = min(max(0, int(smoothed_x)), screen_width - 1)
                        smoothed_y = min(max(0, int(smoothed_y)), screen_height - 1)

                        # Set mouse position
                        win32api.SetCursorPos((int(smoothed_x), int(smoothed_y)))

                        # Update previous coordinates
                        previous_x, previous_y = smoothed_x, smoothed_y

                elif not all(finger_states):  # Not all fingers are open
                    pyautogui.click(button="left")

        # Display the output
        cv2.imshow("Hand Gesture Click System", image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release video capture object and close display windows
video.release()
cv2.destroyAllWindows()
