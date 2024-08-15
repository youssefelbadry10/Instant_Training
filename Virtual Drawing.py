import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand and Drawing Utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define colors for the palette
color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0), (128, 0, 128)]

# Desired display size
display_width = 1280
display_height = 720

# Initialize video capture
video = cv2.VideoCapture(0)

# Set the resolution of the video capture
video.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

# Initialize the Hand Tracker
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    # Initialize previous coordinates for drawing
    previous_x, previous_y = None, None
    selected_color = (0, 255, 0)  # Default color (green)

    # Create a canvas to keep the drawing
    canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Draw color palette
        palette_height = 50
        for i, color in enumerate(color_palette):
            x = i * (display_width // len(color_palette))
            cv2.rectangle(frame, (x, 0), (x + (display_width // len(color_palette)), palette_height), color, -1)

        # Convert the BGR image to RGB before processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hand landmarks
        results = hands.process(image_rgb)

        # Convert the image color back so it can be displayed
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

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
                pixel_coordinates = mp_drawing._normalized_to_pixel_coordinates(
                    index_finger_tip.x, index_finger_tip.y, display_width, display_height
                )

                if pixel_coordinates:
                    current_x, current_y = pixel_coordinates

                    # Check if the index finger is over the color palette
                    if current_y < palette_height:
                        color_index = current_x // (display_width // len(color_palette))
                        selected_color = color_palette[color_index]
                        cv2.putText(image_bgr, f"Color: {selected_color}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 255, 255), 2)
                        previous_x, previous_y = None, None
                    else:
                        # Draw on the canvas if the index finger is detected
                        if previous_x is not None and previous_y is not None:
                            cv2.line(canvas, (previous_x, previous_y), (current_x, current_y), selected_color, 5)

                        # Update previous coordinates
                        previous_x, previous_y = current_x, current_y

                else:
                    # Reset previous coordinates if the finger is not detected
                    previous_x, previous_y = None, None

        # Overlay the drawing canvas on the video feed
        final_output = cv2.addWeighted(image_bgr, 1.0, canvas, 1.0, 0)

        # Display the output with virtual drawing
        cv2.imshow("Virtual Drawing with Hand Gestures", final_output)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release video capture object and close display windows
video.release()
cv2.destroyAllWindows()
