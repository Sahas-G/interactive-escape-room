import time
import pyautogui

class gameState:
    #this class will be used to ensure we enable the appropriate gestures in the given context

    stateModes = ["Menu", "Gameplay", "Inventory", "Diary"]

class keyboardController:
    keyMap = {
        "forward": 'w',
        "backward": 's',
        "right": 'd',
        "left": 'a',
        "inventory": 'i',
        "diary": 'j',
        'flashlight': 't',
        "pause": 'esc',
        "pan_up": 'up',
        "pan_down": 'down',
        "pan_right": 'right',
        "pan_left": 'left',
        "enter": 'enter',
        "click": 'q',
        "rClick": 'z'
    }

# False--> Not being pressed, True--> Being Pressed
    keyState = {
        'w': False,
        's': False,
        'd': False,
        'a': False,
        'esc': False,
        'up': False,
        'down': False,
        'right': False,
        'left': False,
        'enter': False,
        'i': False,
        'j': False,
        't': False,
        'q': False,
        'z': False
    }

    navStates = ['forward', 'backward', 'left', 'right']
    panStates = ['pan_up', 'pan_down', 'pan_left', 'pan_right']

    def startKey(self, action):
        pyautogui.keyDown(self.keyMap[action])
        self.keyState[self.keyMap[action]] = True

    def stopKey(self, action):
        pyautogui.keyUp(self.keyMap[action])
        self.keyState[self.keyMap[action]] = False

    def pressKey(self, action):
        self.keyState[self.keyMap[action]] = True
        pyautogui.press(self.keyMap[action])
        self.keyState[self.keyMap[action]] = False

    def processMovement(self, action, mode='NAV'):
        #Look at the state, if the action is new, turn off everything else, and turn on new one.
        # Mode = NAV or PAN

        if mode == 'NAV':
            scope = self.navStates
        else:
            scope = self.panStates

        if action == 'neutral':
            for item in scope:
                self.stopKey(item)

        else:
            for nav in scope:
                if action!= nav:
                    self.stopKey(action)
                else:
                    self.startKey(action)


if __name__ == '__main__':
    print('Start of Code')

    kb = keyboardController()

    #SETUP
    # Initialise OPenCV Model
    #Initialise Speech Recognition
    # Create Class Objects

    time.sleep(5)

    while True:

        # continuous = ["forward", "backward", "right", "left", "pan_down", "pan_up", "pan_right", "pan_left"]
        # for item in continuous:
        #     kb.startKey(item)
        #     time.sleep(0.5)
        #     kb.stopKey(item)
        #
        # # for other_item in ["inventory", "diary", 'flashlight', "pause"]:
        # #     kb.pressKey(other_item)

    #Step1: Look for speech/ gesture Trigger
    #Step2: If there is a trigger Validate the input can be used in the game state
    #Step3: If valid trigger --> Push to keyboard & update game state


        pyautogui.moveTo(100, 200)
        time.sleep(3)
        pyautogui.moveTo(200, 100)
        time.sleep(3)


