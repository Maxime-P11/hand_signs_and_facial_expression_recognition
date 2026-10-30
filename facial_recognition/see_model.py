#I use mediapipe a model trained by google to get the lanmarks of the face of the image
import cv2
import torch
from utils import landmarker_and_result, normalize_landmarks, Camera, get_list_data_face_landmark, FCNN
import os

#landmarker
MODEL_PATH = "face_landmarker.task"


face_landmarker = landmarker_and_result(MODEL_PATH)

camera = Camera()

model = FCNN()
PATH = os.path.join(os.path.dirname(__file__),'modelparam')
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

while camera.IsOpened():

    isOK = camera.calc_frame()
    if not isOK:
        print("error occured with the camera")
        break

    result = camera.get_frame_landmark(face_landmarker)

    if result.face_landmarks!=[]:
        all_pos, z_all, pos_model = get_list_data_face_landmark(result.face_landmarks[0], camera.get_width(), camera.get_height())

        camera.draw_face_landmark(all_pos, z_all)
        
        pos_model = normalize_landmarks(pos_model)

        pred = torch.argmax(model.forward(torch.from_numpy(pos_model))).item()
        
        if pred == 0:
            text_pred = "heureux/happy"
        else:
            text_pred = "triste/sad"
            
        camera.draw_text(text_pred)


    # display frame
    camera.show_frame()
    if cv2.waitKey(1) == 27: #escape key
        break

# release everything
camera.close()
face_landmarker.close()