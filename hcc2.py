import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import cv2
import os
import numpy as np


def sobel_gradient(image):
    # Compute gradient using Sobel operator
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    return magnitude


# Function to preprocess the images
def preprocess(image):
    # Equalize the image
    equalized_image = cv2.equalizeHist(image)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(equalized_image, (3, 3), 0)

    # Normalize the image
    normalized_image = blurred_image / 255.0

    # Compute Sobel gradient
    magnitude = sobel_gradient(normalized_image)

    return magnitude


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = os.listdir(folder_path)
        self.transform = transforms.Compose([transforms.ToTensor()])  # Convert images to PyTorch tensors

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming images are grayscale
        image = preprocess(image)
        image = self.transform(image)

        # Extract the label from the filename or folder structure if applicable
        # Replace this with your actual label extraction logic
        # TODO
        label = int(image_path.split(os.sep)[-2])  # Assumes the label is encoded in the folder name

        return image, label

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':

    # Specify the paths to your train, validation, and test folders
    train_folder = './gnt-jpg/train'
    val_folder = './gnt-jpg/val'
    test_folder = './gnt-jpg/test'

    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_folder)
    val_dataset = CustomDataset(val_folder)
    test_dataset = CustomDataset(test_folder)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)



    x_train_tensor = torch.Tensor(x_train).permute(0, 3, 1, 2)
    y_train_tensor = torch.Tensor(y_train).type(torch.long)
    x_val_tensor = torch.Tensor(x_val).permute(0, 3, 1, 2)
    y_val_tensor = torch.Tensor(y_val).type(torch.long)
    x_test_tensor = torch.Tensor(x_test).permute(0, 3, 1, 2)
    y_test_tensor = torch.Tensor(y_test).type(torch.long)

    # Instantiate the model
    input_channels = x_train_tensor.shape[1]
    num_classes = 3755
    model = SimpleCNN(input_channels, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}')

    # Test the model
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_accuracy = test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2%}')