import torch
import os
import cv2
from utils import FCNN, landmarker_and_result, normalize_landmarks, getoutput, get_list_data_hand_landmark, Camera


PATH = os.path.join(os.path.dirname(__file__),'modelparam')
model = FCNN()

checkpoint = torch.load(PATH)

model.load_state_dict(checkpoint['model_state_dict'])


start = 0.

#landmarker
MODEL_PATH = os.path.join(os.path.dirname(__file__),'hand_landmarker.task')
hand_landmarker = landmarker_and_result(model_path=MODEL_PATH)

camera = Camera()

while True:
	camera.calc_frame()
	result = camera.get_frame_handmark(hand_landmarker)

	if result is not None:
		if result.hand_landmarks!=[]:				
			all_pos, z_all, pos_model = get_list_data_hand_landmark(result.hand_landmarks[0], camera.get_width(), camera.get_height())
			camera.draw_hand_landmark(all_pos, z_all)
			pos_model = normalize_landmarks(pos_model)
			prediction = getoutput(model, torch.from_numpy(pos_model))

			
			match prediction:
				case 0:
					text_draw = "clenched fist / main en poing"
				case 1:
					text_draw = "flat hand / main à plat"
				case 2:
					text_draw = "index finger pointing up / index qui pointe vers le haut"

			
			camera.draw_text(text_draw)

	# display frame
	camera.show_frame()
	if cv2.waitKey(1) == 27: #escape key
		break

# release everything
camera.exit_programm()
hand_landmarker.close()
