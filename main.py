# import time
# import pyautogui
# from gestures import gestureRecognition
# import gameState

from keyboardController import keyboardController
import multiprocessing
from speechRecognition import mainSpeechRecognition
from recognition import recognitionLoop
import zmq
from gameOverlay import Overlay


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

    def gestureProcess(self, keyStateData, puzzleStateData):
        # self.gestureObj = gestureRecognition()
        # self.gestureObj.prepCamera()
        # self.gestureObj.loadModel()
        # self.gestureObj.remainingStuff()

        print("Starting Gesture Recognition Process")

        while True:
            # keyStateData.put(self.gestureObj.findGesture())
            recognitionLoop(keyStateData, puzzleStateData)

    def keyBoardProcess(self, keyStateData):

        print("Starting Keyboard Process")

        while True:
            while not keyStateData.empty():
                self.kb.updateKeyData(keyStateData.get())
            self.kb.executeKeys()

    def speechProcess(self, keyStateData, gameOverlayState):
        print("Start of Speech Recognition Process")
        mainSpeechRecognition(keyStateData, gameOverlayState)

    def unityCommProcess(self, puzzleStateData):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        print("Starting Unity Communicator Process")

        while True:
            #  Wait for next request from client
            message = socket.recv()
            print("Received request: %s" % message)
            puzzleStateData.put(message)
            socket.send(b"Done")

    def gameOverlayProcess(self, gameOverlayState):
        while True:
            if not gameOverlayState.empty():
                state = gameOverlayState.get()
                if state == "help on":
                    overlay = Overlay(gameOverlayState)
                    print("Start of GUI")
                    overlay.show(gameOverlayState)

    def startThreads(self):
        with multiprocessing.Manager() as manager:
            keyStateData = multiprocessing.Queue()
            puzzleStateData = multiprocessing.Queue()
            gameOverlayState = multiprocessing.Queue()

            # process1 = multiprocessing.Process(target=self.gestureProcess, args=(keyStateData, puzzleStateData))
            # process1.start()
            #
            # process2 = multiprocessing.Process(target=self.keyBoardProcess, args=(keyStateData,))
            # process2.start()

            process3 = multiprocessing.Process(target=self.speechProcess, args=(keyStateData, gameOverlayState))
            process3.start()

            # process4 = multiprocessing.Process(target=self.unityCommProcess, args=(puzzleStateData,))
            # process4.start()

            process5 = multiprocessing.Process(target=self.gameOverlayProcess, args=(gameOverlayState,))
            process5.start()

            # process2.join()
            process3.join()
            # process1.join()
            # process4.join()
            process5.join()


if __name__ == '__main__':
    print('Start of Code')
    interactor = interactionClass()
    interactor.startThreads()
