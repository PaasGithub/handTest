import mediapipe as mp
import cv2
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    #print('gesture recognition result: {}'.format(result))
    #print('Gesture type:{}'.format(result.gestures[0]))
    #print('hand type:{}'.format(result.handedness))

    for gesture in result.gestures:
        #print([category.category_name for category in gesture])
        print('Gesture type:{}'.format([category.category_name for category in gesture]))
    
    for handedness in result.handedness:
        #print([category.category_name for category in gesture])
        print('Hand type:{}'.format([category.category_name for category in handedness]))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/45818/Desktop/code/handTest/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

timestamp = 0
with GestureRecognizer.create_from_options(options) as recognizer:
    # Create a VideoCapture object to access the camera (usually 0 for built-in webcam)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        exit()

    while True:
        # Read a frame from the camera
        success, video = cap.read()
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Check if the frame was successfully read
        if not success:
            print("Error: Could not read frame.")
            break

        timestamp += 1
        
        vidFrame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=video
        )

        recognition_result = recognizer.recognize_async(vidFrame, timestamp)

        if recognition_result is not None:
            #annotated_frame = draw_landmarks_on_image(vidFrame, detection_result)
            #print("this is annotated frame")
            continue
        else:
            print("No hand landmarks detected in this frame.")


        # Display the frame in a window
        #cv2.imshow("Live Video Feed", annotated_frame)
        cv2.imshow("Live Video Feed", video)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release the VideoCapture and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

