import torch
from torchvision import transforms
from hcc2 import NeuralNetwork
import cv2

model = NeuralNetwork().to('cuda')
checkpoint = torch.load("./models/hccr-model2.0.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


