import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from handFunctions import draw_landmarks_on_image

# img = cv2.imread("handTest\handImage1.jpg")

# Check if the image was loaded successfully
""" if img is not None:
    # Check if the image has valid dimensions
    if img.shape[0] > 0 and img.shape[1] > 0:
        cv2.imshow("chosen Image", img)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()
    else:
        print("Image dimensions are not valid.")
else:
    print("Failed to load the image.") """

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='C:/Users/45818/Desktop/code/handTest/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
originalimage = mp.Image.create_from_file("C:/Users/45818/Desktop/code/handTest/handImage1.jpg")

#resize image parameters
desired_width = 800  # Replace with your desired width
desired_height = 600  # Replace with your desired height

# Convert the mp.Image to a NumPy array
image_data = originalimage.numpy_view()

# Resize the image using cv2.resize
resized_image_data = cv2.resize(image_data, (desired_width, desired_height))

#resized_image 
image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=resized_image_data
)

#new. just resize before beginning above

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)
#print(detection_result)
# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

cv2.imshow("chosen Image Boxed", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()