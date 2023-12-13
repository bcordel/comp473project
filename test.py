import torch
from hcc2 import NeuralNetwork
from gnt import GNT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNetwork()
# Move model to device
model = model.to(device)

gnt = GNT()
labels = gnt.create_label_dict("./data/gnt/train/")

# Set model to evaluation mode
model.eval()

pic_file = 'C006-3.jpg'

# Select a sample from the dataset
image = Image.open(f'./data/gnt-jpg/train/{pic_file}')
label = labels[pic_file]
label = [ord(char) % 3755 for char in label]


transform = transforms.Compose([
    transforms.ToTensor(),
])

# Apply the transformations, add a batch dimension, and move the image to the right device
image = transform(image).unsqueeze(0).to(device)

# Set the network to evaluation mode
model.eval()

# Pass the image through the network and get the prediction
with torch.no_grad():
    prediction = model(image)

_, predicted_label = torch.max(prediction, 1)

# Move the image and label back to the CPU for visualization
image = image.to('cpu')

# Print the actual label
print(f'Actual label: {label}')

# Print the predicted label
print(f'Predicted label: {predicted_label.item()}')

# Display the image
plt.imshow(image.squeeze(0).squeeze(0), cmap='gray')
plt.show()
