import time
import pyautogui
from gestures import gestureRecognition
from keyboardController import keyboardController
import gameState
import multiprocessing


class interactionClass:
    def __init__(self):
        self.kb = keyboardController()
        self.keyStateData = {
            "forward": False,
            "backward": False,
            "right": False,
            "left": False,
            "inventory": False,
            "diary": False,
            "flashlight": False,
            "pause": False,
            "pan_up": False,
            "pan_down": False,
            "pan_right": False,
            "pan_left": False,
            "enter": False,
            "click": False,
            "rClick": False
        }

    def gestureProcess(self, keyStateData):
        self.gestureObj = gestureRecognition()
        self.gestureObj.prepCamera()
        self.gestureObj.loadModel()
        self.gestureObj.remainingStuff()

        print("Starting Gesture Recognition Process")

        while True:
            keyStateData.put(self.gestureObj.findGesture())

    def keyBoardProcess(self, keyStateData):

        print("Starting Keyboard Process")

        while True:
            if not keyStateData.empty():
                self.kb.updateKeyData(keyStateData.get())
            self.kb.executeKeys()

    def startThreads(self):
        with multiprocessing.Manager() as manager:
            keyStateData = multiprocessing.Queue()

            process1 = multiprocessing.Process(target=self.gestureProcess, args=(keyStateData,))
            process1.start()

            process2 = multiprocessing.Process(target=self.keyBoardProcess, args=(keyStateData,))
            process2.start()

            process2.join()
            process1.join()


if __name__ == '__main__':
    print('Start of Code')
    interactor = interactionClass()
    interactor.startThreads()
