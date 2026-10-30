from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch

import numpy as np

import os
from utils import FCNN

inputs = np.load('landmarks.npy', allow_pickle=True)
targets = np.load('signification.npy', allow_pickle=True)
inputs = np.reshape(inputs, (int(inputs.shape[0]/1431),1431))

targets = np.eye(2)[targets]


N = len(inputs)

indices = np.random.permutation(N)

split = int(0.9 * N)

train_indices = indices[:split]
test_indices = indices[split:]



train_data = torch.from_numpy(inputs[train_indices]).type(torch.float)
train_target = torch.from_numpy(targets[train_indices]).type(torch.float)

test_data = torch.from_numpy(inputs[test_indices]).type(torch.float)
test_target = torch.from_numpy(targets[test_indices]).type(torch.float)




train_dataset = TensorDataset(train_data, train_target)
test_dataset = TensorDataset(test_data, test_target)

#split trainign data sorted fix later

loaders = {
    'train': DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=0
    ),

    'test': DataLoader(
        test_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=0
    )
}


    

#lr= learning rate
model = FCNN()
optimizer= optim.Adam(model.parameters(), lr=1e-4 )

loss_fn = nn.CrossEntropyLoss()


def train():
	model.train()
	for data, target in loaders['train']:
		optimizer.zero_grad()
		output = model(data)
		loss = loss_fn(output, target)
		loss.backward()
		optimizer.step()

def test(epoch):
	model.eval()
	test_loss=0
	correct=0

	with torch.no_grad():
		for data, target in loaders['test']:
			output = model(data)
			test_loss += loss_fn(output, target)
			pred = torch.argmax(output, dim=1, keepdim=True)

			correct+= pred.eq(torch.argmax(target, dim=1, keepdim=True)).sum().item()

	test_loss /= len(loaders['test'].dataset)

	accuracy=correct/len(loaders['test'].dataset)
	print(f"epoch = {epoch}, average loss : {test_loss:.4f}, Accuracy : {accuracy*100:4f}%")
	return accuracy



epoch = 1
bestacc = 0
PATH = os.path.join(os.path.dirname(__file__),'modelparam')
if os.path.exists(PATH):
	checkpoint = torch.load(PATH)

	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	bestacc = checkpoint['bestacc']

	print("loaded saved model")
else:
	print("created new model")


while True:
	train()
	accuracy = test(epoch)
	if accuracy>bestacc:
		bestacc = accuracy
		checkpoint = {
		    'epoch': epoch + 1,
			'bestacc': bestacc,
		    'model_state_dict': model.state_dict(),
		    'optimizer_state_dict': optimizer.state_dict(),
		}

		torch.save(checkpoint, PATH)
		print(f"Checkpoint saved at epoch {epoch} to {PATH}")

		
	epoch+=1
