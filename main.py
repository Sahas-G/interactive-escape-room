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
        "enter": 'enter'
    }

# False--> Not being pressed, True--> Being Pressed
    keyState = {
        'w': False,
        's': False,
        'd': False,
        'a': False,
        'i': False,
        'esc': False,
        'up': False,
        'down': False,
        'right': False,
        'left': False,
        'enter': False,
        'i': False,
        'j': False,
        't': False
    }

    def startKey(self, action):
        pyautogui.keyUp(self.keyMap[action])
        self.keyState[self.keyMap[action]] = True

    def stopKey(self, action):
        pyautogui.keyDown(self.keyMap[action])
        self.keyState[self.keyMap[action]] = False

    def pressKey(self, action):
        self.keyState[self.keyMap[action]] = True
        pyautogui.press(self.keyMap[action])
        self.keyState[self.keyMap[action]] = False

if __name__ == '__main__':
    print('Start of Code')

    kb = keyboardController()

    while True:

        continuous = ["forward", "backward", "right", "left", "pan_up", "pan_down", "pan_right", "pan_left"]
        for item in continuous:
            kb.startKey(item)
            time.sleep(1)
            kb.stopKey(item)

        # for other_item in ["inventory", "diary", 'flashlight', "pause"]:
        #     kb.pressKey(other_item)


