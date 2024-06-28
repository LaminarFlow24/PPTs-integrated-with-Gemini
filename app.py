#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import time
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import time
import numpy as np


import PIL.Image
import numpy as np
import google.generativeai as genai

#API key for google generative ai
genai.configure(api_key="")


import cv2 as cv2
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier



segmentor = SelfiSegmentation()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=1)########################################### external camera chosen as default... put 0 for webcam
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=536)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    
    args = get_args()
    indexImg = 0
    start_time = None
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    ww, hh = 960, 536
    prevx, prevy = 0,0
    lis = [0,0,0,0]

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera start ###############################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2, ################################################### 
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()


    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    

    # FPS ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    

    

    #  ########################################################################
    mode = 0

    ################################################## Adding background images
    listImg = os.listdir("BackgroundImages")
    imgList = []
    for imgPath in listImg:
        img = cv2.imread(f'BackgroundImages/{imgPath}')
        imgList.append(img)

    ############################## Masking for annotations

    mask = np.ones((hh, ww))*255 ##############
    mask = mask.astype('uint8')


    

    while True:
        fps = cvFpsCalc.get()

        #################################################
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)
        if key == ord('c'):
            mask = np.ones((hh,ww))*255 ##############
            mask = mask.astype('uint8')
        elif key == ord('a') and indexImg > 0:
            indexImg -= 1
        elif key == ord('d') and indexImg < len(imgList) - 1:
            indexImg += 1
        

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)  # Mirror display ####################################
        debug_image = copy.deepcopy(image)
        uimage = copy.deepcopy(image)
        
        # Detection implementation #############################################################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        debug_image = backadd(uimage,imgList,indexImg)
        
        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Writing to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list, lis)

                is_right_hand = handedness.classification[0].label == 'Right'
                
                
                # Hand signs
                
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                xi, yi = int(hand_landmarks.landmark[8].x * ww), int(hand_landmarks.landmark[8].y * hh)
                
                if hand_sign_id == 2 and is_right_hand:  # Point gesture
                    
                    cv2.line(mask, (prevx, prevy), (xi, yi), 0, 6)
                    prevx, prevy = xi, yi
                    
                else:   
                    prevx, prevy = xi,yi

                if hand_sign_id == 3 and is_right_hand:
                    if start_time is None:
                        start_time = time.time()
                        
                    elif time.time() - start_time >= 1:
                        sendtogem(gem_image)


                        
                        folder_name = 'gemini_sent'
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        
                        
                        file_path = os.path.join(folder_name, 'processed_image.png')

                        gem_image = cv2.cvtColor(gem_image, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(file_path, cv2.cvtColor(gem_image, cv2.COLOR_RGB2BGR))

                        start_time = None
                
                    

                elif hand_sign_id == 4 and is_right_hand == False:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 1:
                        if indexImg < len(imgList) - 1:
                            indexImg += 1
                            start_time = None
                

                elif hand_sign_id == 5 and is_right_hand == False:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 1:
                        if indexImg > 0:
                            indexImg -= 1
                            start_time = None
                

                elif hand_sign_id == 6 and is_right_hand == False:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 1:
                        mask = np.ones((hh,ww))*255 ##############
                        mask = mask.astype('uint8')
                        start_time = None
                    
                else:
                    start_time = None

                

                # Drawing part
                
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    
                )
        else:
            pass

        

        
        debug_image = draw_info(debug_image, fps, mode, number)

        # for drawing annotation #########################
        op = cv2.bitwise_and(debug_image, debug_image, mask=mask)
        debug_image[:, :, 1] = op[:, :, 1]
        debug_image[:, :, 2] = op[:, :, 2]

        

        cv2.imshow('Hand Gesture Recognition', debug_image)
        
        gem_image = debug_image  #photo sent to gemini
        
        



    cap.release()
    cv2.destroyAllWindows()




def backadd(img,imgList,indexImg):
    imgOut = segmentor.removeBG(img, imgList[indexImg], cutThreshold=0.2)
    
    # Get the dimensions of the image
    h, w, _ = img.shape

    # Create a mask for the right half of the image
    maskk = np.zeros_like(img)
    maskk[:, 0:w//2] = img[:, 0:w//2]

    # Combine the original left half with the segmented right half
    combined = np.where(maskk != 0, maskk, imgOut)

    return combined


def sendtogem(image):
    hei, wid,_ = image.shape
    start_col = wid // 2
    end_col = wid
    right_half = image[:, start_col:end_col]
    f_image = cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB)

    # Create the folder if it doesn't exist
    folder_name = 'gemini_sent'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Define the file path
    file_path = os.path.join(folder_name, 'processed_image.png')
    
    # Save the image
    cv2.imwrite(file_path, cv2.cvtColor(f_image, cv2.COLOR_RGB2BGR))

    imgarr = PIL.Image.fromarray(f_image)
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(["What is dark blue line pointing at? Describe it in detail.", imgarr], stream=True)
    # Initialize an empty string to store the final text
    generated_text = ""

    for content in response:
        # Append each generated chunk to the final text
        generated_text += content.text

    print(generated_text)
    print("###############################################")


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list, lis):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        if lis[number-4] <= 1500:
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            lis[number-4] += 1
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        else:
            print("STOP")

    
    return

#for drawing lines over hand
def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)


    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # if finger_gesture_text != "":
    #     cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    #     cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv2.LINE_AA)

    return image


# def draw_point_history(image, point_history):
#     for index, point in enumerate(point_history):
#         if point[0] != 0 and point[1] != 0:
#             cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),
#                       (152, 251, 152), 2)

#     return image


def draw_info(image, fps, mode, number):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)

    mode_string = ['Logging Key Point']
    if mode == 1:
        cv2.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv2.LINE_AA)
    return image


if __name__ == '__main__':
    main()
