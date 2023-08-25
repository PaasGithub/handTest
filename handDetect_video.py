import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from handFunctions import draw_landmarks_on_image


""" # Create a VideoCapture object to access the camera (usually 0 for built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Read a frame from the camera
    success, vidframe = cap.read()

    # Check if the frame was successfully read
    if not success:
        print("Error: Could not read frame.")
        break

    # Display the frame in a window
    cv2.imshow("Live Video Feed", vidframe)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows() """

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))
    #print(".")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/45818/Desktop/code/handTest/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

with HandLandmarker.create_from_options(options) as landmarker:
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
        
        vidFrame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=video
        ) 
        
        #print ("this is vidframe:",vidFrame)

        #Detect hand landmarks from the live feed.
        detection_result = landmarker.detect_async(vidFrame, frame_timestamp_ms)
        
        print("this is;",detection_result)
        #Process the classification result. In this case, visualize it.
        #annotated_frame = draw_landmarks_on_image(vidFrame, detection_result)

        # if detection_result is not None:
        #     #annotated_frame = draw_landmarks_on_image(vidFrame, detection_result)
        #     print("this is annotated frame")
        # else:
        #     print("No hand landmarks detected in this frame.")

        # Display the frame in a window
        #cv2.imshow("Live Video Feed", annotated_frame)
        cv2.imshow("Live Video Feed", video)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release the VideoCapture and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
