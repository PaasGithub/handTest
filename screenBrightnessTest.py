import screen_brightness_control as sbc
import pyautogui

def brightTest():
    #get current brightness of primary display (monitor)
    current_brightness = sbc.get_brightness(display=0)

    #print current brightness of primary display (monitor)
    #comes as integer
    print("Original current brightness: ", current_brightness)
    #print("This is get brightness function type: ", type(current_brightness))

    #if brightness is less than 50 increase to 100
    #if brightness is more than 50 decrease to 9

    #brightness comes as a list
    #convert to integer to use with operators
    brightness_str = ''.join(map(str, current_brightness))
    brightness_int = int(brightness_str)

    if (int(brightness_int) < 50):
        sbc.set_brightness(100, display=0)

        #get new current brightness of primary display (monitor)
        new_current_brightness = sbc.get_brightness(display=0)

        #print current brightness of primary display (monitor)
        #comes as integer
        print("New current brightness: ", new_current_brightness)
    else:
        sbc.set_brightness(9, display=0)

        #get new current brightness of primary display (monitor)
        new_current_brightness = sbc.get_brightness(display=0)

        #print current brightness of primary display (monitor)
        #comes as integer
        print("New current brightness: ", new_current_brightness)

def volumeTest():
    x = 50
    a = (x+2)
    print (a)
    pyautogui.press('volumeup',2)

volumeTest()