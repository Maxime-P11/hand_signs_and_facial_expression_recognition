I decided to share a repo for those two programm since they share a lot of code but still required some work to go from one to the other, but they are still independant from one to each other you can only download one if you want.

facial_recognition :</br>
&emsp;&emsp;&emsp;- face_landmarker.task : google's model to create a face_mesh</br>

&emsp;&emsp;&emsp;- landmarks.npy : dataset where all the face_mesh are stored</br>

&emsp;&emsp;&emsp;- signification.npy : dataset where all the signification for the model are stored 0 for happy and 1 for sad</br>

&emsp;&emsp;&emsp;-> landmarks.npy and signification.npy are preloaded with a dataset I created from my face, you can delete them and camera.py will create new ones</br>
&emsp;&emsp;&emsp;- camera.py : used to add to the dataset by pressing 1 for happy and 2 for sad on the keyboard (not the numpad)  and by making the coresponding face, only save face mesh not the images.</br>
&emsp;&emsp;&emsp;- model_trainer.py : trains a model on the dataset created by camera.py using pytorch and save the model parameters and other things to modelparam. you can quit and resume the programm and it will save it whenever there is a new best on the test dataset.
To restart training simply delete modelparam.</br>
&emsp;&emsp;&emsp;- modelparam : saved parameters of the model in this format : {</br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    'epoch': epoch ,</br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    'bestacc': bestacc,</br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    'model_state_dict': model.state_dict(),</br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    'optimizer_state_dict': optimizer.state_dict(),</br>
&emsp;&emsp;&emsp;}</br>

&emsp;&emsp;&emsp;- utils.py  : some function and classes common to the programms</br>
&emsp;&emsp;&emsp;- see_model.py : a programm to see the model in action on your wecam where it predict your emotion either happy or sad in real time</br>
&emsp;&emsp;&emsp;-> for all programm except model_trainer.py you can hold the key "esc" to quit the programm</br>


hand_recognition :</br>
&emsp;&emsp;&emsp;- face_landmarker.task : google's model to calculate hand landmark</br>
&emsp;&emsp;&emsp;- landmarks.npy : dataset where all the hand landmarks are stored</br>
&emsp;&emsp;&emsp;- signification.npy : dataset where all the signification for the model are stored 0 for clenched fist, 1 for flat hand and 2 for index finger pointing up</br>
&emsp;&emsp;&emsp;-> landmarks.npy and signification.npy are preloaded with a dataset I created from my hand, you can delete them and camera.py will create new ones, since I used only my right hand for the training if your left handed it might perform worse</br>
&emsp;&emsp;&emsp;- camera.py : used to add to the dataset by pressing 0 for clenched fist, 1 for flat hand and 2 for sad on the keyboard (not the numpad)  and by making the coresponding hand gesture, only save the hand landmarks not the images.</br>
&emsp;&emsp;&emsp;- model_trainer.py : trains a model on the dataset created by camera.py using pytorch and save the model parameters and other things to modelparam. you can quit and resume the programm and it will save it whenever there is a new best on the test dataset.
To restart training simply delete modelparam.</br>
&emsp;&emsp;&emsp;- modelparam : saved parameters of the model in this format : {</br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    'epoch': epoch ,</br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    'bestacc': bestacc,</br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    'model_state_dict': model.state_dict(),</br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    'optimizer_state_dict': optimizer.state_dict(),</br>
&emsp;&emsp;&emsp;}</br>

&emsp;&emsp;&emsp;- utils.py  : some function and classes common to the programms</br>
&emsp;&emsp;&emsp;- see_model.py : a programm to see the model in action on your wecam where it predict your hand gesture</br>
&emsp;&emsp;&emsp;-> for all programm except model_trainer.py you can hold the key "esc" to quit the programm</br>
