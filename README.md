# 6.835 Final Project: Interactive Multimodal Room Escape Game

## Contents

|**File**| **Description** |
|--|--|
|**ROOT**|
|main.py| The code starts here. It creates and manages all the multiprocessing threads|
|requirements.txt| List of all the 3rd party packages that are needed for this project to run|
|**ASSETS**|
|GameV10.zip|The Unity game --> Needs to be unzipped. The "Room Escape.exe" file inside needs to be executed to launch the game|
|Action Recognition Training Module.ipynb| Jupiter notebook used for training the LSTM model for action recognition|
|PyAudio-0.2.11-cp310-cp310-win_amd64.whl| PyAudio library for Python 3.10 for windows x64 systems|
|**PROCESSES**|
|gameOverlay.py| Contains the tkinter code to create a GUI overlay on top of the game, used as the help menu|
|instructions.png| image that is displayed in GUI help overlay|
|keyboardController.py| It executes key presses that are added to the keyStateData queue by the other processes. Key presses are singular continuous until stopped, or finite such as 20 key presses.|
|PROCESSES/**GESTURE RECOGNITION**|
|recognition.py| Gesture Recognition process --> looks for hand poses, actions, creates dynamic hot zones for navigation and directly manipulates the mouse in puzzle mode.|
|model_rh_only.h5|Action recognition model file|
|PROCESSES/**SPEECH**|
|speechRecognition.py| Speech Recognition process looks for pre-determined keywords, and takes corresponding actions such as triggering a key press and opening the help overlay. |
|room-escape-speech.json| Google Cloud Speech to Text API credentials file|



## Setup
**This project will run using Windows OS (10+), Python 3.10 and a laptop with integrated webcam**
We highly recommend running this project on a dual screen setup.
The laptop (with the webcam) will be the secondary screen, and the external monitor is the **primary/ main** screen.

### Installation
 1. download this github repository.
 2. install all the packages mentioned in requirements.txt | `pip install -r requirements.txt`
 3. You will need to separately install the PyAudio package from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio), we have also included the exact  library we used inside the assets folder.
 4. Download and unzip the GameV10.zip file

### Launch
 1. Execute main.py
 2. Wait for the webcam output window to pop up.
 3. Connect the external display. **[This step is important, connect external display only after webcam output window is visible]**
 4. Move the webcam output window to the secondary laptop screen.
 5. Execute the "RoomEscape.exe" file to launch the game.
 6. Say "Help", and familiarize yourself with the gestures and commands. Say "Close/ Go away" after you're done reading it.
 7. Try navigating the room using hand gestures. See the Laptop display with the webcam output for feedback on how the system is responding in real time.

### Troubleshooting
 1. You can reduce the speed of panning by going to the in game pause menu --> options --> inputs --> reduce mouse sensitivity 
 2.  If you are unable to move forward, there may be furniture in your way. Use "step right", "step left", "step back" and "look back" to maneuver out of the tight spot. 
 3. If you are in a neutral position with a fist and you are not moving forward, open your palm and make a fist again
