import cv2
import mediapipe as mp

# Initialize MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object for your camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame to RGB (MediaPipe uses RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)  # Get the number of detected hands
        print(f"Number of hands detected: {num_hands}")

        for landmarks in results.multi_hand_landmarks:
            # Loop through landmarks and draw them on the frame
            for landmark in landmarks.landmark:
                x, y, z = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]), int(landmark.z * frame.shape[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with hand landmarks
    cv2.imshow('Hand Landmarks', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Release MediaPipe Hands
hands.close()
