#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clash Royale Gesture Control

Control Clash Royale using hand gestures detected through your webcam.
Uses MediaPipe for hand tracking and a custom TensorFlow Lite model for gesture classification.
"""

import sys
import os
import csv
import copy
import argparse
import itertools
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui as pag

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# Default gesture mappings
GESTURE_ACTION_MAP_LEFT = {
    # Card Selection
    "one_finger_point": "select_card_1",
    "two_finger_point": "select_card_2", 
    "three_fingers_up": "select_card_3",
    "four_fingers_up": "select_card_4",
    
    # Navigation & Reset
    "close_palm": "cancel_selection",
    
    # Emotes
    "thumbs_down": "emote_oops",
    "thumbs_up": "emote_thanks", 
    "ok": "emote_well_played",
}

GESTURE_ACTION_MAP_RIGHT = {
    # Placement & Confirmation
    "one_finger_point": "drag_card",
    "open_palm": "deploy",
    "fist": "wait"
}

# Default screen configuration
DEFAULT_CAM_W = 1280
DEFAULT_CAM_H = 720
DEFAULT_SCREEN_BOX_COORDS = [[676, 585], [1095, 25]]
DEFAULT_CR_SCREEN_COORDS = (1000, 75)
DEFAULT_CR_SCREEN_WIDTH = 510
DEFAULT_CR_SCREEN_HEIGHT = 900
DEFAULT_ACTION_BOX_WIDTH = 415
DEFAULT_ACTION_BOX_HEIGHT = 550


class Controller:
    """Game controller that translates gestures to game actions."""
    
    def __init__(self, screen_config):
        self.action_num_l = 0
        self.action_num_r = 0
        self.screen_config = screen_config
        
    def route(self, action_num, handedness, **kwargs):
        """Route gesture to appropriate action based on hand."""
        if handedness == 'Right':
            if action_num == 2:
                pass
            elif action_num == self.action_num_r or action_num in [3,4] and self.action_num_r in [3,4]:
                return
            self.action_num_r = action_num
            if self.action_num_r == 0:
                self.deploy()
            elif self.action_num_r == 2:
                self.drag(**kwargs)

        elif handedness == 'Left':
            if action_num == self.action_num_l:
                return 
            self.action_num_l = action_num
            if self.action_num_l == 2:
                self.select_card(1)
            elif self.action_num_l == 5:
                self.select_card(2)
            elif self.action_num_l == 6:
                self.select_card(3)
            elif self.action_num_l == 7:
                self.select_card(4)
            elif self.action_num_l == 4:
                self.emote("Thumbs up")
            elif self.action_num_l == 8:
                self.emote("Thumbs down")
            elif self.action_num_l == 3:
                self.emote("Well played")
        
    def select_card(self, card_num):
        """Select a card by pressing the corresponding number key."""
        pag.press(str(card_num))
        
    def deploy(self):
        """Deploy the selected card by releasing the mouse."""
        print("Deploying")
        pag.mouseUp(button='left')
    
    def emote(self, emote_name):
        """Trigger an emote in the game."""
        pag.press('e')
        
        if emote_name == 'Thumbs up':
            pag.press('w')
        elif emote_name == 'Thumbs down':
            pag.press('s')
        elif emote_name == 'Well played':
            pag.press('a')
    
    def drag(self, **kwargs):
        """Drag a card to position it on the battlefield."""
        coords = kwargs['index_tip']
        x = coords.x * self.screen_config['cam_w'] - self.screen_config['screen_box_coords'][0][0]
        y = -coords.y * self.screen_config['cam_h'] + self.screen_config['screen_box_coords'][0][1]
        x = self._clip(x, 0, self.screen_config['action_box_width'])
        y = self._clip(y, 0, self.screen_config['action_box_height'])
        
        print(f"Drag position: ({x}, {y})")
        
        cr_x = (self.screen_config['cr_screen_coords'][0] + 
                x / self.screen_config['action_box_width'] * self.screen_config['cr_screen_width'])
        cr_y = (self.screen_config['cr_screen_coords'][1] + self.screen_config['cr_screen_height'] - 
                y / self.screen_config['action_box_height'] * self.screen_config['cr_screen_height'])
        
        pag.mouseDown(cr_x, cr_y)
    
    def activate_skill(self):
        """Activate a skill or special ability."""
        pag.press('e')

    def _clip(self, x, low, high):
        """Clamp value between low and high."""
        return min(max(x, low), high)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Clash Royale Gesture Control - Control the game using hand gestures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Camera settings
    parser.add_argument('--device', type=int, default=0,
                       help='Camera device number')
    parser.add_argument('--width', type=int, default=int(960 * 0.7),
                       help='Camera capture width')
    parser.add_argument('--height', type=int, default=int(540 * 0.7),
                       help='Camera capture height')
    
    # MediaPipe settings
    parser.add_argument('--use-static-image-mode', action='store_true',
                       help='Use static image mode for MediaPipe')
    parser.add_argument('--min-detection-confidence', type=float, default=0.7,
                       help='Minimum confidence for hand detection')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.3,
                       help='Minimum confidence for hand tracking')
    parser.add_argument('--max-num-hands', type=int, default=2,
                       help='Maximum number of hands to detect')
    
    # Screen configuration
    parser.add_argument('--cam-resolution-w', type=int, default=DEFAULT_CAM_W,
                       help='Camera resolution width for coordinate mapping')
    parser.add_argument('--cam-resolution-h', type=int, default=DEFAULT_CAM_H,
                       help='Camera resolution height for coordinate mapping')
    
    # Display options
    parser.add_argument('--show-bounding-rect', action='store_true', default=True,
                       help='Show bounding rectangles around hands')
    parser.add_argument('--show-landmarks', action='store_true', default=True,
                       help='Show hand landmarks')
    
    # Model settings
    parser.add_argument('--model-path', type=str, 
                       default='model/keypoint_classifier/keypoint_classifier.tflite',
                       help='Path to the gesture classification model')
    parser.add_argument('--labels-path', type=str,
                       default='model/keypoint_classifier/keypoint_classifier_label.csv',
                       help='Path to the gesture labels file')
    
    return parser.parse_args()


def select_mode(key, mode):
    """Select data collection mode based on key press."""
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    """Calculate bounding rectangle for hand landmarks."""
    image_width, image_height = image.shape[1], image.shape[0]
    
    landmark_array = np.empty((0, 2), int)
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    """Calculate landmark positions in image coordinates."""
    image_width, image_height = image.shape[1], image.shape[0]
    
    landmark_point = []
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point


def pre_process_landmark(landmark_list):
    """Preprocess landmarks for model input."""
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Convert to one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    
    def normalize_(n):
        return n / max_value
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list


def pre_process_point_history(image, point_history):
    """Preprocess point history for model input."""
    image_width, image_height = image.shape[1], image.shape[0]
    
    temp_point_history = copy.deepcopy(point_history)
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    
    # Convert to one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    
    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    """Log data for training purposes."""
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])


def draw_landmarks(image, landmark_point):
    """Draw hand landmarks on the image."""
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0 or index == 1:  # Wrist
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        elif index in [2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:  # Joints
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        elif index in [4, 8, 12, 16, 20]:  # Fingertips
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    """Draw bounding rectangle around detected hand."""
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    """Draw gesture information text on the image."""
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "" and finger_gesture_text is not None:
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    return image


def draw_info(image, fps, mode, number):
    """Draw FPS and mode information on the image."""
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def main():
    """Main function to run the gesture control system."""
    args = get_args()
    
    print("🎮 Clash Royale Gesture Control")
    print("=" * 50)
    print(f"Camera device: {args.device}")
    print(f"Camera resolution: {args.width}x{args.height}")
    print(f"Detection confidence: {args.min_detection_confidence}")
    print(f"Tracking confidence: {args.min_tracking_confidence}")
    print(f"Max hands: {args.max_num_hands}")
    print("=" * 50)
    print("Controls:")
    print("  ESC - Exit")
    print("  k   - Enter gesture logging mode")
    print("  n   - Normal mode")
    print("=" * 50)
    
    # Screen configuration
    screen_config = {
        'cam_w': args.cam_resolution_w,
        'cam_h': args.cam_resolution_h,
        'screen_box_coords': DEFAULT_SCREEN_BOX_COORDS,
        'cr_screen_coords': DEFAULT_CR_SCREEN_COORDS,
        'cr_screen_width': DEFAULT_CR_SCREEN_WIDTH,
        'cr_screen_height': DEFAULT_CR_SCREEN_HEIGHT,
        'action_box_width': DEFAULT_ACTION_BOX_WIDTH,
        'action_box_height': DEFAULT_ACTION_BOX_HEIGHT,
    }
    
    # Initialize camera
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize game controller
    controller = Controller(screen_config)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=0
    )
    
    # Initialize gesture classifier
    keypoint_classifier = KeyPointClassifier()
    
    # Load gesture labels
    with open(args.labels_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    
    print(f"Loaded {len(keypoint_classifier_labels)} gesture classes:")
    for i, label in enumerate(keypoint_classifier_labels):
        print(f"  {i}: {label}")
    print("=" * 50)
    
    # Initialize FPS calculator
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    
    # Initialize history for point tracking
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    
    mode = 0
    
    print("Starting gesture recognition... Press ESC to exit.")
    
    while True:
        fps = cvFpsCalc.get()
        
        # Process key input
        key = cv.waitKey(10)
        if key == 27:  # ESC
            print("Exiting...")
            break
        number, mode = select_mode(key, mode)
        
        # Capture frame
        ret, image = cap.read()
        if not ret:
            continue
        
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        
        # Convert to RGB for MediaPipe
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        
        # Process detected hands
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calculate bounding box and landmarks
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # Preprocess landmarks for classification
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                
                # Log data if in logging mode
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)
                
                # Classify gesture
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                # Prepare data for controller
                kwargs = {}
                hand_label = handedness.classification[0].label
                if hand_label == "Right":
                    kwargs['index_tip'] = hand_landmarks.landmark[8]
                
                # Route to game controller
                controller.route(hand_sign_id, hand_label, **kwargs)
                
                # Draw visualization
                if args.show_bounding_rect:
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                if args.show_landmarks:
                    debug_image = draw_landmarks(debug_image, landmark_list)
                
                debug_image = draw_info_text(
                    debug_image, brect, handedness,
                    keypoint_classifier_labels[hand_sign_id], None
                )
        else:
            point_history.append([0, 0])
        
        # Draw info and screen box
        debug_image = draw_info(debug_image, fps, mode, number)
        cv.rectangle(debug_image, screen_config['screen_box_coords'][0], 
                    screen_config['screen_box_coords'][1], (255, 0, 0), 5)
        
        # Display the image
        cv.imshow('Clash Royale Gesture Control', debug_image)
    
    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    print("Gesture control stopped.")


if __name__ == '__main__':
    main() 