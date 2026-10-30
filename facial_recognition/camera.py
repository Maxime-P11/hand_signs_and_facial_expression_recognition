#I use mediapipe a model trained by google to get the lanmarks of the face of the image
import cv2
import os
import numpy as np

from utils import landmarker_and_result, normalize_landmarks, Camera, get_list_data_face_landmark

#landmarker
MODEL_PATH = "face_landmarker.task"

face_landmarker = landmarker_and_result(MODEL_PATH)

camera = Camera()

PATH_LANDMARK = os.path.join(os.path.dirname(__file__),'landmarks.npy')
PATH_SIGNIFICATION = os.path.join(os.path.dirname(__file__),'signification.npy')

#assume PATH_LANDMARK and PATH_SIGNIFICATIONare in sinc and it should
if not os.path.exists(PATH_LANDMARK):
    all_data = np.array([])
    all_signification = np.array([], dtype=int)
else:
    all_data = np.load('landmarks.npy', allow_pickle=True)
    all_signification = np.load('signification.npy', allow_pickle=True)



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

        key = cv2.waitKey(1)
        if 49<=key<=50:
            all_data = np.append(all_data, pos_model)
            all_signification = np.append(all_signification, key-49)

            unique, counts = np.unique(all_signification, return_counts=True)
            print(counts)

        

             
    # display frame
    camera.show_frame()
    if cv2.waitKey(1) == 27: #escape key
        break

# release everything
np.save('landmarks.npy', all_data, allow_pickle=True)
np.save('signification.npy', all_signification, allow_pickle=True)
        
camera.close()
face_landmarker.close()