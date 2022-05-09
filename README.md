## 6.835 Final Project: Interactive Multimodal Room Escape Game

## Contents

|File| Description |
|--|--|
|main.py| The code starts here. It creates and manages all the multiprocessing threads|
|recognition.py| Gesture Recognition process --> looks for hand poses, actions, creates dynamic hot zones for navigation and directly manipulates the mouse in puzzle mode.|
|model_rh_only.h5|Action recognition model file|
|keyboardController.py| It executes key presses that are added to the keyStateData queue by the other processes. Key presses are singular continuous until stopped, or finite such as 20 key presses.|
|speechRecognition.py| Speech Recognition process looks for pre-determined keywords, and takes corresponding actions such as triggering a key press and opening the help overlay. |
|room-escape-speech.json| Google Cloud Speech to Text API credentials file|
|gameOverlay.py| Contains the tkinter code to create a GUI overlay on top of the game, used as the help menu|
|instructions.png| image that is displayed in GUI help overlay|
|requirements.txt| List of all the 3rd party packages that are needed for this project to run|
