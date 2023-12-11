import torch
from torchvision import transforms
from hcc2 import NeuralNetwork, ChineseDataset
import cv2

model = NeuralNetwork().to('cuda')
checkpoint = torch.load("./models/hccr-model2.0.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


transforms = transforms.ToTensor()

image_path = "./data/unit-tests/C002-0.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

image = ChineseDataset.preprocess(image)

input_data = torch.stack([transforms(image) for _ in range (64)]).to('cuda')

with torch.no_grad():
    output = model(input_data)

_, predicted = torch.max(output.data, 1)

print(f"Predicted: {predicted.tolist()}")

