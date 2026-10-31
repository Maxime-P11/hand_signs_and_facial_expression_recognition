import torch
import os
import cv2
import pyautogui
import keyboard
import mouse

from utils import FCNN, landmarker_and_result, normalize_landmarks, getoutput, get_list_data_hand_landmark, Camera



PATH = os.path.join(os.path.dirname(__file__),'modelparam')
model = FCNN()

checkpoint = torch.load(PATH)

model.load_state_dict(checkpoint['model_state_dict'])


#landmarker
MODEL_PATH = os.path.join(os.path.dirname(__file__),'hand_landmarker.task')
hand_landmarker = landmarker_and_result(model_path=MODEL_PATH)

camera = Camera()

WINDOW_WIDTH = 3500
WINDOW_HEIGHT = int(3500*(camera.get_height()/camera.get_width()))
mouseright, mouseleft = False, False

while True:
	camera.calc_frame()
	result = camera.get_frame_handmark(hand_landmarker)

	if result is not None:
		if result.hand_landmarks!=[]:				
			all_pos, z_all, pos_model = get_list_data_hand_landmark(result.hand_landmarks[0], camera.get_width(), camera.get_height())

			camera.draw_hand_landmark(all_pos, z_all)

			pos_model = normalize_landmarks(pos_model)
			
			prediction=getoutput(model, torch.from_numpy(pos_model))

			camera.draw_text(str(prediction))

			if prediction == 0:
				if mouseright:
					pyautogui.mouseUp(button='right')
					mouseright = False
				if mouseleft:
					pyautogui.mouseUp(button='left')
					mouseleft = False
				

				mouse.move(int(result.hand_landmarks[0][0].x*WINDOW_WIDTH-700), int(result.hand_landmarks[0][0].y*WINDOW_HEIGHT-700))
				

			elif prediction == 2:
				pyautogui.mouseDown(button='right')
				mouseright = True
			elif prediction == 1:
				pyautogui.mouseDown(button='left')
				mouseleft = True
				
			

			#1 boule
			#2 main plate doight écarté
			#3 index pointé


	camera.show_frame()
	if keyboard.is_pressed('esc'):
		break
	if cv2.waitKey(1) == 27: #escape key
		break
	if cv2.getWindowProperty('frame',cv2.WND_PROP_VISIBLE) < 1:        
		break 

# release everything
camera.exit_programm()
hand_landmarker.close()