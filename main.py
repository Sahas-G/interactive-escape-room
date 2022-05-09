from keyboardController import keyboardController
import multiprocessing
from speechRecognition import mainSpeechRecognition
from recognition import recognitionLoop
import zmq
from gameOverlay import Overlay


class interactionClass:
    """
    This class has the function definitions for the infinite loops of the 5 processes.
    Each process function is first initiliased if applicable and then executes on an
    infinite loop.
    The startThreads() creates the multiprocesses.
    """
    def gestureProcess(self, keyStateData, puzzleStateData):
        print("Starting Gesture Recognition Process")
        while True:
            recognitionLoop(keyStateData, puzzleStateData)

    def keyBoardProcess(self, keyStateData):
        print("Starting Keyboard Process")
        kb = keyboardController()
        while True:
            while not keyStateData.empty():
                kb.updateKeyData(keyStateData.get())
            kb.executeKeys()

    def speechProcess(self, keyStateData, gameOverlayState):
        print("Starting Speech Recognition Process")
        mainSpeechRecognition(keyStateData, gameOverlayState)

    def unityCommProcess(self, puzzleStateData):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        print("Starting Unity Communicator Process")
        while True:
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
        """
        Function creates the below listed processes:
        1. gestureProcess --> Gesture Recognition looks for hand poses, 
                actions, creates dynamic hot zones and directly manipulates 
                the mouse in puzzle mode.
        2. keyBoardProcess --> It executes key presses that are added to the
                keyStateData queue by the other processes. key presses are singular
                continuous until stopped, or finite such as 20 key presses.
        3. speechProcess --> Speech Recognition process looks for pre-determined
                keywords, and takes corresponding actions such as triggering a key
                press and opening the help overlay.
        4. unityCommProcess --> This process acts as a server for a websocket
                connection. The Unity application is modified to send out websocket
                requests to this specific address to communicate with the python app.
        5. gameOverlayProceess --> When speech recognition process deetects "help", this,
                triggers the gameoverlay process, which open a tkinter based GUI window
                to display the help menu.
        """
        keyStateData = multiprocessing.Queue()
        puzzleStateData = multiprocessing.Queue()
        gameOverlayState = multiprocessing.Queue()

        process1 = multiprocessing.Process(target=self.gestureProcess, args=(keyStateData, puzzleStateData))
        process1.start()

        process2 = multiprocessing.Process(target=self.keyBoardProcess, args=(keyStateData,))
        process2.start()

        process3 = multiprocessing.Process(target=self.speechProcess, args=(keyStateData, gameOverlayState))
        process3.start()

        process4 = multiprocessing.Process(target=self.unityCommProcess, args=(puzzleStateData,))
        process4.start()

        process5 = multiprocessing.Process(target=self.gameOverlayProcess, args=(gameOverlayState,))
        process5.start()

        process2.join()
        process3.join()
        process1.join()
        process4.join()
        process5.join()


if __name__ == '__main__':
    print('Start of Multimodal Room Escape Experience')
    interactor = interactionClass()
    interactor.startThreads()
