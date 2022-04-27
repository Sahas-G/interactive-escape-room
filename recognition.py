import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import math

import tensorflow as tf
from tensorflow import keras

# draw the detected images to the screen
mp_drawing = mp.solutions.drawing_utils
# import mediapipe holistic model
mp_holistic = mp.solutions.holistic

wCam, hCam = 640, 480
frameReduction = 100
smoothingThreshold = 5
wScr, hScr = pyautogui.size()  # Outputs the high and width of the screen
# plocX, plocY = 0, 0
threshold = 20
recognition_threshold = 0.7

# Defined as top left to bottom right of the zone
hotZones = {
    "left": [(0, 0), (wScr / 3, hScr)],
    "right": [(wScr * 2 / 3, 0), (wScr, hScr)],
    "up": [(0, 0), (wScr, hScr / 3)],
    "down": [(0, hScr * 3 / 5), (wScr, hScr)],
    "neutral": [(wScr / 3, hScr / 3), (wScr * 2 / 3, hScr * 2 / 3)]
}

Keys = {
            "forward": None,
            "backward": None,
            "right": None,
            "left": None,
            "inventory": None,
            "diary": None,
            "flashlight": None,
            "pause": None,
            "pan_up": None,
            "pan_down": None,
            "pan_right": None,
            "pan_left": None,
            "enter": None,
            "click": None,
            "rClick": None,
            "focus": None,
            "run": None,
            "look_back": None
        }

actions = np.array(['move', 'select', 'click', 'flip', 'grab'])
sequence = []

def inZones(x, y):
    """
    check if the current mouse position is in a hot zone

    :param x: x coordinate
    :param y: y coordinate
    :return: a list of the hot zones that the mouse has been in
    """
    zoneList = []
    for zone_name in hotZones.keys():
        zone = hotZones[zone_name]
        if zone[0][0] <= x < zone[1][0]:
            if zone[0][1] <= y < zone[1][1]:
                zoneList.append(zone_name)
    return zoneList


def scaling(original_value, original_max, original_min, scaled_max, scaled_min):
    """
    scale a value from one range to another

    :param original_value: value to be scaled
    :param original_max: original max value of the range that the value is in
    :param original_min: original min value of the range that the value is in
    :param scaled_max: max value of the desired range
    :param scaled_min: min value of the desired range
    :return:
    """

    original_range = (original_max - original_min)
    scaled_range = (scaled_max - scaled_min)
    scaled_value = (((original_value - original_min) * scaled_range) / original_range) + scaled_min

    return scaled_value


def calc_euclidean_distance(x1, y1, x2, y2):
    """
    calculate euclidean distance between two points

    :param x1: point 1 x coordinate
    :param y1: point 1 y coordinate
    :param x2: point 2 x coordinate
    :param y2: point 2 y coordinate
    :return: euclidean distance between two points
    """
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return math.sqrt(dist)


def mediapipe_detection(image, model):
    """
    takes in a frame in OpenCV, process it with a MediaPipe model
    return a processed frame

    :param image: every frame in OpenCV
    :param model: what MediaPipe model is used (we use Holistic)
    :return:
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_hand_landmarks(image, results):
    """

    :param image: every frame in OpenCV
    :param results: resutls after frame (image) is processed by MediaPipe Holistic Model
    :return: N/A
    """
    # Draw body pose landmarks
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
    #                           mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    #                           )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_hand_keypoints(results):
    """
    extract landmark points from pose & hand MediaPipe model

    :param results: frames after processed by MediaPipe model
    :return: cancatenated numpy array
    """
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # return np.concatenate([pose, face, lh, rh])
    return np.concatenate([lh, rh])

def navigation_recognition(results, plocX, plocY):
    """
    detect if there's a right hand in the processed frames (results)
    if there's a right hand, parse all the landmarks and update the mouse position based on index finger location
    :param results: frames processed by MediaPipe model
    :param plocX: previous x location = 0
    :param plocY:
    :return: a list of zones that nav

    FUTURE IMPROVEMENT:
    not just detect hand, but detect specific gestures, e.g. having only one index finger
    """

    landmarkList = []
    zoneList = []
    returnX, returnY = plocX, plocY

    if results.right_hand_landmarks is not None:
        # len(results.right_hand_landmarks.landmark) = 21
        # there are 21 landmarks in right_hand_landmarks with 3 coordinates (x,y,z) for each landmark
        # see a detailed image of all 21 landmarks here: https://google.github.io/mediapipe/solutions/hands.html
        for res in results.right_hand_landmarks.landmark:
            # iterate through all 21 landmarks
            landmarkList.append([res.x, res.y])

    if len(landmarkList) != 0:
        x1, y1 = landmarkList[8]  # the 9th element (index:8) in the landmarkList is always INDEX_FINGER_TIP
        scaledX = scaling(x1, 1, 0, wScr, 0)
        scaledY = scaling(y1, 1, 0, hScr, 0)

        if calc_euclidean_distance(scaledX, scaledY, plocX, plocY) > threshold:
            pyautogui.moveTo(wScr - scaledX, scaledY)
            returnX = scaledX
            returnY = scaledY

        zoneList = inZones(wScr - scaledX, scaledY)

    return zoneList, returnX, returnY


def action_recognition(results, model, threshold):

    keypoints = extract_hand_keypoints(results)
    sequence.append(keypoints)
    current_sequence = sequence[-30:]

    if len(current_sequence) == 30:
        res = model.predict(np.expand_dims(current_sequence, axis=0))[0]
        # print(actions[np.argmax(res)])
        if res[np.argmax(res)] > threshold:
            # print("returning it!")
            return actions[np.argmax(res)]

    return ""

def walk_recognition(results):

    if results.left_hand_landmarks is not None:

        thumb_finger_tip = results.left_hand_landmarks.landmark[4]
        index_finger_tip = results.left_hand_landmarks.landmark[8]
        middle_finger_tip = results.left_hand_landmarks.landmark[12]
        ring_finger_tip = results.left_hand_landmarks.landmark[16]
        pinky_finger_tip = results.left_hand_landmarks.landmark[20]

        if thumb_finger_tip and index_finger_tip and middle_finger_tip and ring_finger_tip and pinky_finger_tip is not None:
            return "move"
    return ""

    # Step 1 : Check if each finger is open or bent.
    #
    # Step 2: If all fingers are bent or all non-thumb fingers are bent, clasify as fist.


def recognitionLoop(keyStateData):

    # loading recognition model
    model = keras.models.load_model("hand_model.h5")

    cap = cv2.VideoCapture(0)
    # VideoCapture(0), 0 means the default camera for our PC/laptop
    # if we have several cameras, the number refers to USB port number.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

    # initialize empty sequence, used to hold the most recent 30 sequences for action recognition
    sequence = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # increase min_detection_confidence & min_tracking_confidence for better accuracy

        # while video capture device is open
        plocX, plocY = 0, 0
        oldZones = []
        oldActivity = ''
        while cap.isOpened():
            # capture frame by frame
            # red is a boolean, indicating if a frame is read correctly

            tmpKeyData = Keys.copy()

            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)
            draw_hand_landmarks(image, results)

            zones, plocX, plocY = navigation_recognition(results, plocX, plocY)

            if not (zones == oldZones):
                if len(zones) >0:
                    print("New Data")
                    for item in zones:
                        if item == "left":
                            tmpKeyData["pan_left"] = "continuous"
                        if item == "right":
                            tmpKeyData["pan_right"] = "continuous"
                        if item == "up":
                            tmpKeyData["pan_up"] = "continuous"
                        if item == "down":
                            tmpKeyData["pan_down"] = "continuous"
                        if item == "neutral":
                            tmpKeyData["pan_left"] = False
                            tmpKeyData["pan_right"] = False
                            tmpKeyData["pan_up"] = False
                            tmpKeyData["pan_down"] = False

                        for pan in ["pan_left", "pan_right", "pan_up", "pan_down"]:
                            if tmpKeyData[pan] is None:
                                tmpKeyData[pan] = False

                else:
                    print("Stop Data")
                    tmpKeyData["pan_left"] = False
                    tmpKeyData["pan_right"] = False
                    tmpKeyData["pan_up"] = False
                    tmpKeyData["pan_down"] = False
                    # tmpKeyData["forward"] = False

                # keyStateData.put(tmpKeyData)
                oldZones = zones

            # activity = action_recognition(results, model, recognition_threshold)
            activity = walk_recognition(results)
            if not (activity == oldActivity):
                if len(activity) > 0:
                    # print("New Data")
                    if activity == "move":
                        tmpKeyData["forward"] = "continuous"
                        # print("Moving")
                    # if activity == "select":
                    #     tmpKeyData["click"] = "single"
                    #     print("clicked")
                    # if activity == "click":
                    #     tmpKeyData["click"] = "single"
                    # if activity == "flip":
                    #     tmpKeyData["click"] = "single"
                    # if activity == "grab":
                    #     tmpKeyData["click"] = "single"
                        pass

                else:
                    print("Stop Data")
                    # tmpKeyData["pan_left"] = False
                    # tmpKeyData["pan_right"] = False
                    # tmpKeyData["pan_up"] = False
                    # tmpKeyData["pan_down"] = False
                    tmpKeyData["forward"] = False

                # keyStateData.put(tmpKeyData)
                oldActivity = activity

            keyStateData.put(tmpKeyData)


            cv2.imshow('Webcam Feed w. hand & pose detection', cv2.flip(image, 1))
            # name the frame, and render the image that we just processed (and flip it to mirror us)

            if cv2.waitKey(10) & 0xFF == ord('.'):
                # we can hit "q" to exit out the frame
                break

    cap.release()
    cv2.destroyAllWindows()
