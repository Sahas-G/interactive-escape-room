# 6.835 Final Project: Interactive Multimodal Room Escape Game

## Contents

|File| Description |
|--|--|
|main.py| Main file to execute - creates and manages threads for multiprocessing |
|recognition.py| Gesture recognition using MediaPipe which recognizes hand poses, actions, and creates dynamic hot zones for navigation and directly manipulates the mouse in puzzle mode. |
|model_rh_only.h5| Our trained model that recognizes actions |
|keyboardController.py| Executes key presses that are added to the keyStateData queue by the other processes. Key presses are singular continuous until stopped, or finite such as 20 key presses.|
|speechRecognition.py| Speech recognition process looks for pre-determined keywords, and takes corresponding actions such as triggering a key press and opening the help overlay. |
|room-escape-speech.json| Google Cloud Speech to Text API credentials file|
|gameOverlay.py| Contains the tkinter code to create a GUI overlay on top of the game, used as the help menu|
|instructions.png| image that is displayed in GUI help overlay|
|requirements.txt| List of all the 3rd party packages that are needed for this project to run|
|GameV10.zip|The Unity game --> Needs to be unzipped. The "Room Escape.exe" file inside needs to be executed to launch the game|

## Setup
**This project will run using Windows OS (10+), needs Python 3.9+ and a laptop with integrated webcam**
We highly recommend running this project on a dual screen setup.
The laptop (with the webcam) will be the secondary screen, and the external monitor is the primary main screen.

Please follow the below steps to launch this project:
 1. download this github repository.
 2. install all the packages mentioned in requirements.txt `pip install -r requirements.txt`
 3. You may need to separately install the PyAudio package from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) in case of PyAudio install error in step 2.
 4. Execute main.py
 5. Wait for the webcam output window to pop up.
 6. Connect the external display.
 7. Move the webcam output window to the secondary laptop screen.
 8. Launch the Unity Game (See steps below)
 9. Download and unzip the GameV10.zip file 
 10. Execute the "RoomEscape.exe" file to launch the game.
