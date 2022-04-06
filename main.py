import time
import pyautogui
from gestures import gestureRecognition
import keyboardController
import gameState


if __name__ == '__main__':
    print('Start of Code')

    kb = keyboardController()

    # SETUP
    # Initialise OPenCV Model
    gestureObj = gestureRecognition(kb)
    # Initialise Speech Recognition

    while True:
        gestureObj.findGesture()
