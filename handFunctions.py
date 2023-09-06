#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

import screen_brightness_control as sbc

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

class displayLandmarks:
  def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]
      handedness = handedness_list[idx]

      # Draw the hand landmarks.
      hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

      # Get the top left corner of the detected hand's bounding box.
      height, width, _ = annotated_image.shape
      x_coordinates = [landmark.x for landmark in hand_landmarks]
      y_coordinates = [landmark.y for landmark in hand_landmarks]
      text_x = int(min(x_coordinates) * width)
      text_y = int(min(y_coordinates) * height) - MARGIN

      # Draw handedness (left or right hand) on the image.
      cv2.putText(annotated_image, f"{handedness[0].category_name}",
                  (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

class gestureAssign:

  def incBrightness(detected_gesture, chosen_gesture):
    if ( detected_gesture == chosen_gesture):
      #get current brightness of primary display (monitor)
      current_brightness = sbc.get_brightness(display=0)

      #print current brightness of primary display (monitor)
      #comes as list
      print("Original current brightness: ", current_brightness)

      brightness_str = ''.join(map(str, current_brightness))
      brightness_int = int(brightness_str)

      if (brightness_int < 100):
        newBrightness_int = brightness_int + 1
        sbc.set_brightness(newBrightness_int, display=0)

        #get new current brightness of primary display (monitor)
        new_current_brightness = sbc.get_brightness(display=0)

        print("New current brightness: ", new_current_brightness)    

 #def incVolume(detected_gesture, chosen_gesture):
        
