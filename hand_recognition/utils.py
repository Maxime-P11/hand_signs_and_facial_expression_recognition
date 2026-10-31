import torch.nn as nn
import torch.nn.functional as F

import mediapipe as mp
from time import time

import numpy as np
import cv2

import torch


class FCNN(nn.Module):
	
	def __init__(self):
		super(FCNN, self).__init__()

		self.fc1 = nn.Linear(60,128)
		self.fc2 = nn.Linear(128,64)
		self.fc3 = nn.Linear(64,32)
		self.fc4 = nn.Linear(32,3)

	def forward(self, x):
		x = F.relu(self.fc1(x))

		x = F.relu(self.fc2(x))

		x = F.relu(self.fc3(x))

		x = self.fc4(x)

		return F.softmax(x, dim=0)


class landmarker_and_result:
    def __init__(self, model_path):
        self.result = None
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.model_path = model_path
        self.createLandmarker()
        

    def createLandmarker(self):

        def update_result(
            result: mp.tasks.vision.HandLandmarkerResult,
            output_image: mp.Image,
            timestamp_ms: int
        ):
            self.result = result

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            result_callback=update_result
        )

        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        self.landmarker.detect_async(
            image=mp_image,
            timestamp_ms=int(time() * 1000)
        )

    def close(self):
        self.landmarker.close()



def display(positions, depths, image):
	data = np.array(depths)
	depths = (data - np.min(data))*255 / (np.max(data) - np.min(data))
	for index, (posxy, depth) in enumerate(zip(positions, depths)):
		if not index in (0, 5, 9, 13, 17):
			x_old = round(positions[index-1][0])
			y_old = round(positions[index-1][1])
		elif index == 17:
			x_old = round(positions[0][0])
			y_old = round(positions[0][1])

		z_color = (max(0, depth),max(0, depth),max(0, depth))

		if not index in (0, 5, 9, 13):
			image = cv2.line(image, posxy, (x_old, y_old), z_color, 1)
		image = cv2.circle(image, posxy, 3, z_color, -1)
	return image


def normalize_landmarks(landmarks_2dlist):
	points = np.array(landmarks_2dlist, dtype=np.float32)
	
	# Translation invariance
	points = points - points[0]
	points = points[1:]
	# Scale invariance
	scale = np.linalg.norm(points[9])

	if scale > 1e-6:
		points = points / scale

	return points.flatten()


def getoutput(model, data):
	model.eval()

	with torch.no_grad():
		output = model(data)
		pred = torch.argmax(output).item()

	return pred


def get_list_data_hand_landmark(hand_data, img_width, img_height):
	all_pos = []
	z_all = []
	pos_model = []

	for hand_landmark in hand_data:
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
		_, self.frame = self.cap.read()
		self.frame = cv2.flip(self.frame, 1)

	def get_frame_handmark(self, handmarker:landmarker_and_result):
		handmarker.detect_async(self.frame)
		
		return handmarker.result

	def draw_hand_landmark(self, all_pos, z_all):
		self.frame = display(all_pos, z_all, self.frame)

	def draw_text(self, text:str, color = (255, 0, 0)):
		self.frame = cv2.putText(self.frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

	def show_frame(self, title="frame"):
		cv2.imshow(title, self.frame)

	def exit_programm(self):
		self.cap.release()
		cv2.destroyAllWindows()
