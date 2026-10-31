import torch
import numpy as np
import os
import cv2
from utils import FCNN, landmarker_and_result, normalize_landmarks, Camera, get_list_data_hand_landmark


PATH = os.path.join(os.path.dirname(__file__),'modelparam')
model = FCNN()


checkpoint = torch.load(PATH)

model.load_state_dict(checkpoint['model_state_dict'])


#landmarker
MODEL_PATH = os.path.join(os.path.dirname(__file__),'hand_landmarker.task')
hand_landmarker = landmarker_and_result(model_path=MODEL_PATH)

camera = Camera()

PATH_LANDMARK = os.path.join(os.path.dirname(__file__),'landmarks.npy')
PATH_SIGNIFICATION = os.path.join(os.path.dirname(__file__),'signification.npy')


if not os.path.exists(PATH_LANDMARK):
    all_data = np.array([])
    all_signification = np.array([], dtype=int)
else:
    all_data = np.load('landmarks.npy', allow_pickle=True)
    all_signification = np.load('signification.npy', allow_pickle=True)

while True:
	camera.calc_frame()
	result = camera.get_frame_handmark(hand_landmarker)

	if result is not None:
		if result.hand_landmarks!=[]:				
			all_pos, z_all, pos_model = get_list_data_hand_landmark(result.hand_landmarks[0], camera.get_width(), camera.get_height())

			camera.draw_hand_landmark(all_pos, z_all)

			pos_model = normalize_landmarks(pos_model)
			key = cv2.waitKey(1)
			if 49<=key<=51: #key 1 to 3 on the keyboard (not numpad)

				all_data = np.append(all_data, pos_model)
				all_signification = np.append(all_signification, key-49)

				unique, counts = np.unique(all_signification, return_counts=True)
				print(counts)

				#1 boule
				#2 main plate doight écarté
				#3 index pointé

	# display frame
	camera.show_frame()
	if cv2.waitKey(1) == 27: #escape key
		break

# release everything
np.save('landmarks.npy', all_data, allow_pickle=True)
np.save('signification.npy', all_signification, allow_pickle=True)

camera.exit_programm()
hand_landmarker.close()
