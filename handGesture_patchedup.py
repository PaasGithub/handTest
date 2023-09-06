import mediapipe as mp
import cv2
from handFunctions import draw_landmarks_on_image
import threading 
import screen_brightness_control as sbc

class GestureRecognizer:
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

    # Create a gesture recognizer instance with the live stream mode:
    def print_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.lock.acquire() # solves potential concurrency issues
        self.current_gestures = []
        
        for gesture in result.gestures:
            #print([category.category_name for category in gesture])
            print('Gesture type:{}'.format([category.category_name for category in gesture]))
            self.current_gestures.append([category.category_name for category in gesture])

            if (([category.category_name for category in gesture]) == ['Pointing_Up']):
                #get current brightness of primary display (monitor)
                current_brightness = sbc.get_brightness(display=0)

                #print current brightness of primary display (monitor)
                #comes as integer
                print("Original current brightness: ", current_brightness)

                brightness_str = ''.join(map(str, current_brightness))
                brightness_int = int(brightness_str)


                if (brightness_int < 100):
                    newBrightness_int = brightness_int + 1
                    sbc.set_brightness(newBrightness_int, display=0)

                    #get new current brightness of primary display (monitor)
                    new_current_brightness = sbc.get_brightness(display=0)

                    print("New current brightness: ", new_current_brightness)
            else:
                continue
        
        for handedness in result.handedness:
            #print([category.category_name for category in gesture])
            print('Hand type:{}'.format([category.category_name for category in handedness]))

        self.lock.release()


    def main(self):
        self.lock = threading.Lock()
        self.current_gestures = []

        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        #GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='C:/Users/45818/Desktop/code/handTest/gesture_recognizer.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands = 2,
            result_callback=self.print_result)

        timestamp = 0
        with GestureRecognizer.create_from_options(options) as recognizer:
            # Initialize MediaPipe Hand Landmarker
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands()
            mp_drawing = mp.solutions.drawing_utils

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
                #print(timestamp)

                # Convert the frame to RGB (MediaPipe uses RGB images)
                frame_rgb = cv2.cvtColor( video, cv2.COLOR_BGR2RGB)

                # Process the frame and detect hands
                detectresults = hands.process(frame_rgb)

                if detectresults.multi_hand_landmarks:
                    num_hands = len(detectresults.multi_hand_landmarks)  # Get the number of detected hands
                    #print(f"Number of hands detected: {num_hands}")

                    vidFrame = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=video
                    )
                    #print(vidFrame)

                    recognition_result = recognizer.recognize_async(vidFrame, timestamp)

                    for landmarks in detectresults.multi_hand_landmarks:
                        # Loop through landmarks and draw them on the frame
                        for landmark in landmarks.landmark:
                            x, y, z = int(landmark.x * video.shape[1]), int(landmark.y * video.shape[0]), int(landmark.z * video.shape[1])
                            cv2.circle(video, (x, y), 5, (0, 255, 0), -1)
                            #cv2.putText(video,())
                        # Draw hand landmarks on the frame
                        mp_drawing.draw_landmarks(video, landmarks, mp_hands.HAND_CONNECTIONS)
                
                    self.put_gestures(video)

                #else:
                    #print("noHands")

                # Display the frame in a window
                #cv2.imshow("Live Video Feed", annotated_frame)
                cv2.imshow("Live Video Feed", video)

                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            # Release the VideoCapture and close the OpenCV windows
            cap.release()
            cv2.destroyAllWindows()

    def put_gestures(self,frame):
        self.lock.acquire()
        gestures = self.current_gestures
        self.lock.release()
        y_pos = 50
        for hand_gesture_name in gestures:
            #print("This is put gesture output:", hand_gesture_name)
            # show the prediction on the frame
            cv2.putText(frame, str(hand_gesture_name), (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            y_pos += 50

if __name__ == "__main__":
    rec = GestureRecognizer()
    rec.main()


#had help from here https://tinyurl.com/gestrureRec