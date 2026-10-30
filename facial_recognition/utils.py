import torch.nn as nn
import torch.nn.functional as F

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from time import time

import numpy as np
import cv2

import torch


class FCNN(nn.Module):
    
	def __init__(self):
		super(FCNN, self).__init__()

		self.fc1 = nn.Linear(1431,256)
		self.fc2 = nn.Linear(256,128)
		self.fc3 = nn.Linear(128,32)
		self.fc4 = nn.Linear(32,2)

	def forward(self, x):
		x = F.relu(self.fc1(x))

		x = F.relu(self.fc2(x))

		x = F.relu(self.fc3(x))

		x = self.fc4(x)

		return F.softmax(x, dim=0)


class landmarker_and_result:
    def __init__(self,
                  model_path,
                  num_face=1,
                  min_detection_confidence = 0.5,
                  min_tracking_confidence = 0.5,
                  min_presence_confidence = 0.5):


        self.model_path = model_path
        base_options = python.BaseOptions(model_asset_path = model_path)
        options = vision.FaceLandmarkerOptions(
            base_options = base_options,
            running_mode = vision.RunningMode.VIDEO,
            num_faces = num_face,
            min_face_detection_confidence = min_detection_confidence,
            min_face_presence_confidence = min_presence_confidence                      ,
            min_tracking_confidence = min_tracking_confidence
        )        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.timestamp_begin = time()


    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, int((time() - self.timestamp_begin)*1000))
        return result

    def close(self):
        self.landmarker.close()




def display_landmarks(all_pos, z_all, img):
    z_all_np = np.array(z_all)
    #multiply by coef to amplify color, and muliply by negative to invert color
    z_all_np *= -10
    #tanh func then put in [0, 255] interval to determine color
    z_all_np = np.tanh(z_all_np)
    z_all_np = (z_all_np + 1) * (255/2)

    for index, pos in enumerate(all_pos):
        color = z_all_np[index]
        color = (color, color, color)

        if index == 4:
            color = (1 * color[0],0,0)

        if index == 168:
            color = (0, 0, 1 * color[0])

        cv2.circle(img, pos, 2, color, -1)
    return img


def normalize_landmarks(landmarks_2dlist):
    points = np.array(landmarks_2dlist, dtype=np.float32)
    
    # Translation invariance
    points = points - points[4]
    points = np.delete(points, (4), axis=0)
    # Scale invariance
    vec_scale = np.copy(points[168])
    scale = (np.sum(np.square(vec_scale)))**(1/3)
    if scale > 1e-6:
        points = points / scale

    return points.flatten()


def getoutput(model, data):
    model.eval()

    with torch.no_grad():
        output = model(data)
        pred = torch.argmax(output).item()

    return pred


def get_list_data_face_landmark(landmark_data, img_width, img_height):
    all_pos = []
    z_all = []
    pos_model = []

    for hand_landmark in landmark_data:
        pos_model.append([hand_landmark.x, hand_landmark.y, hand_landmark.z])
        x_pos = round(hand_landmark.x * img_width)
        y_pos = round(hand_landmark.y * img_height)
    
        all_pos.append((x_pos, y_pos))
    
        z_all.append(hand_landmark.z)

    return all_pos, z_all, pos_model


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.img_width  = self.cap.get(3)
        self.img_height = self.cap.get(4)
        self.frame = None

    def get_width(self):
        return self.img_width
    def get_height(self):
        return self.img_height

    def calc_frame(self):
        isOK, self.frame = self.cap.read()
        self.frame = cv2.flip(self.frame, 1)
        return isOK

    def get_frame_landmark(self, handmarker:landmarker_and_result):
        return handmarker.process_frame(self.frame)

    def draw_face_landmark(self, all_pos, z_all):
        self.frame = display_landmarks(all_pos, z_all, self.frame)

    def draw_text(self, text:str, color = (255, 0, 0)):
        self.frame = cv2.putText(self.frame, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def show_frame(self, title="frame"):
        cv2.imshow(title, self.frame)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def IsOpened(self):
        return self.cap.isOpened()