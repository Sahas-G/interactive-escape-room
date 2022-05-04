import pyautogui


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
        "rClick": 'z',
        "focus": 'g',
        "run": 'shiftleft',
        "look_back": 'k'
    }

    # continuous, single, false
    # This structure is updated continuously
    oldKeyState = {
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
        "rClick": False,
        "focus": False,
        "run": False,
        "look_back": False
    }

    currentKeyState = {
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
        "rClick": False,
        "focus": False,
        "run": False,
        "look_back": False
    }

    navStates = ['forward', 'backward', 'left', 'right']
    panStates = ['pan_up', 'pan_down', 'pan_left', 'pan_right']

    def __init__(self):
        pyautogui.FAILSAFE = False

    def updateKeyData(self, newKeyData):
        self.oldKeyState = self.currentKeyState.copy()

        for iter in newKeyData.keys():
            if newKeyData[iter] is None:
                pass
            else:
                self.currentKeyState[iter] = newKeyData[iter]

        for action in self.oldKeyState.keys():

            if self.oldKeyState[action] == "continuous" and self.currentKeyState[action] is False:
                self.stopKey(action)
                print("Stopped %s" % action)

    """
    Goes through list of all the keys. If state is single, press key once and set the state to default of False
    If state is continuous, press key and don't change the state
    """

    def executeKeys(self):
        for action in self.currentKeyState.keys():
            if not self.currentKeyState[action]:
                pass
            elif self.currentKeyState[action] == "single":
                pyautogui.press(self.keyMap[action])
                self.currentKeyState[action] = False
                print("Executed: %s" % action)
            elif self.currentKeyState[action] == "continuous":
                self.startKey(action)
            elif type(self.currentKeyState[action]) == int:
                if self.currentKeyState[action] == 1:
                    self.stopKey(action)
                else:
                    self.currentKeyState[action] -= 1
                    self.startKey(action)



    """
    startKey cannot do continuous key presses, limited usage.
    Deprecated functions below.
    """

    def startKey(self, action):
        pyautogui.keyDown(self.keyMap[action])
        # self.currentKeyState[action] = True

    def stopKey(self, action):
        pyautogui.keyUp(self.keyMap[action])
        self.currentKeyState[action] = False


    def pressKey(self, action):
        self.currentKeyState[action] = True
        pyautogui.press(action)
        self.currentKeyState[action] = False

    def processMovement(self, action, mode='NAV'):
        # Look at the state, if the action is new, turn off everything else, and turn on new one.
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
                if action is not nav:
                    if self.currentKeyState[action] is True:
                        self.stopKey(action)
                else:
                    if not self.currentKeyState[action]:
                        self.startKey(action)
