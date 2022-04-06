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
                    self.stopKey(action)
                else:
                    if not self.keyState[self.keyMap[action]]:
                        self.startKey(action)