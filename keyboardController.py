import pyautogui


class keyboardController:
    """
    Each key has 4 states:
    Single --> The key is pressed once
    Continuous --> They key is pressed continuously until stopped.
    False --> The key is not to be pressed, or stop being pressed.
    int --> The key is pressed int times.
    """

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

    def executeKeys(self):
        """
        Goes through list of all the keys. 
            If state is single, press key once and set the state to False
            If state is continuous, press key and don't change the state. 
            If state is integer, press and decrement counter. If counter is 1 set to False.
        """
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

    def startKey(self, action):
        pyautogui.keyDown(self.keyMap[action])

    def stopKey(self, action):
        pyautogui.keyUp(self.keyMap[action])
        self.currentKeyState[action] = False

    def pressKey(self, action):
        self.currentKeyState[action] = True
        pyautogui.press(action)
        self.currentKeyState[action] = False
