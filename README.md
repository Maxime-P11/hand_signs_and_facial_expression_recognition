I decided to share a repo for those two programm since they share a lot of code but still required some work to go from one to the other, but they are still independant from one to each other you can only download one if you want.

facial_recognition :</br>
&ensp;&ensp;&ensp;- face_landmarker.task : google's model to create a face_mesh</br>
&ensp;&ensp;&ensp;- landmarks.npy : dataset where all the face_mesh are stored</br>
  	- signification.npy : dataset where all the signification for the model are stored 0 for happy and 1 for sad</br>
  	-> landmarks.npy and signification.npy are preloaded with a dataset I created from my face, you can delete them and camera.py will create new ones</br>
  	- camera.py : used to add to the dataset by pressing 1 for happy and 2 for sad on the keyboard (not the numpad)  and by making the coresponding face, only save face mesh not the images.</br>
  	- model_trainer.py : trains a model on the dataset created by camera.py using pytorch and save the model parameters and other things to modelparam. you can quit and resume the programm and it will save it whenever there is a new best on the test dataset.
To restart training simply delete modelparam.</br>
  	- modelparam : saved parameters of the model in this format : {</br>
		    'epoch': epoch ,</br>
			'bestacc': bestacc,</br>
		    'model_state_dict': model.state_dict(),</br>
		    'optimizer_state_dict': optimizer.state_dict(),</br>
    }</br>

  	- utils.py  : some function and classes common to the programms</br>
  	- see_model.py : a programm to see the model in action on your wecam where it predict your emotion either happy or sad in real time</br>
  	-> for all programm except model_trainer.py you can hold the key "esc" to quit the programm</br>


hand_recognition :</br>
  	- face_landmarker.task : google's model to calculate hand landmark</br>
  	- landmarks.npy : dataset where all the hand landmarks are stored</br>
  	- signification.npy : dataset where all the signification for the model are stored 0 for clenched fist, 1 for flat hand and 2 for index finger pointing up</br>
  	-> landmarks.npy and signification.npy are preloaded with a dataset I created from my hand, you can delete them and camera.py will create new ones, since I used only my right hand for the training if your left handed it might perform worse</br>
  	- camera.py : used to add to the dataset by pressing 0 for clenched fist, 1 for flat hand and 2 for sad on the keyboard (not the numpad)  and by making the coresponding hand gesture, only save the hand landmarks not the images.</br>
  	- model_trainer.py : trains a model on the dataset created by camera.py using pytorch and save the model parameters and other things to modelparam. you can quit and resume the programm and it will save it whenever there is a new best on the test dataset.
To restart training simply delete modelparam.</br>
  	- modelparam : saved parameters of the model in this format : {</br>
		    'epoch': epoch ,</br>
			'bestacc': bestacc,</br>
		    'model_state_dict': model.state_dict(),</br>
		    'optimizer_state_dict': optimizer.state_dict(),</br>
    }</br>

  	- utils.py  : some function and classes common to the programms</br>
  	- see_model.py : a programm to see the model in action on your wecam where it predict your hand gesture</br>
  	-> for all programm except model_trainer.py you can hold the key "esc" to quit the programm</br>
