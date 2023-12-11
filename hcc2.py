import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import cv2
import os
import numpy as np
import time


torch.cuda.empty_cache()
torch.cuda.device('cuda:0')


def create_label_dict(dir_name):
    label_dict = {}

    for file in os.listdir(dir_name):
        file_name = os.path.join(dir_name, file)
        num = 0
        try:
            with open(file_name, 'rb') as image_file:
                # Get length of file
                file_length = os.path.getsize(file_name)

                # While current cursor spot is less than length
                while image_file.tell() < file_length:
                    # Skip length of image (we get this from w x h)
                    image_file.read(4)

                    # Get label
                    label = image_file.read(2)

                    # Get image dimensions
                    width = int.from_bytes(image_file.read(2), byteorder='little')
                    height = int.from_bytes(image_file.read(2), byteorder='little')

                    # Get byte array of gray-scale image
                    image_file.read(width * height)

                    # Save label
                    entry_name = f"{os.path.basename(file_name)[:4]}-{num}.jpg"
                    label_dict[entry_name] = label.decode("GBK")
                    num += 1

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
    return label_dict


class ChineseDataset(Dataset):
    def __init__(self, folder_path, label_dict):
        self.folder_path = folder_path
        self.image_files = os.listdir(folder_path)
        self.label_dict = label_dict
        self.transform = transforms.Compose([transforms.ToTensor()])  # Convert images to PyTorch tensors

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = preprocess(image)
        image = self.transform(image)
        label = self.label_dict[self.image_files[idx]]
        label = [ord(char) % 3755 for char in label]
        label = torch.tensor(label)

        return image, label

    def sobel_gradient(image):
        # Compute gradient using Sobel operator
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        return magnitude

    # Function to preprocess the images
    def preprocess(self, image):
        # Equalize the image
        equalized_image = cv2.equalizeHist(image)

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(equalized_image, (3, 3), 0)

        # Normalize the image
        normalized_image = blurred_image / 255.0

        # Compute Sobel gradient
        magnitude = image.sobel_gradient(normalized_image)

        return magnitude


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1/3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(50, 100, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1/3),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(100, 150, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1/3),
            nn.Dropout(p=0.1),
            nn.Conv2d(150, 200, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1/3),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(200, 250, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1/3),
            nn.Dropout(p=0.2),
            nn.Conv2d(250, 300, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1/3),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(1200, 1600),
            nn.LeakyReLU(negative_slope=1/3),
            nn.Dropout(p=0.5),
            nn.Linear(1600, 900),
            nn.LeakyReLU(negative_slope=1/3),
            nn.Linear(900, 200),
            nn.LeakyReLU(negative_slope=1/3),
            nn.Linear(200, 3755)
        )

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
                layer.weight.data = layer.weight.data.float()
                layer.bias.data = layer.bias.data.float()

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    # Paths to train, validation, and test jpg folders
    train_jpg_folder = './data/gnt-jpg/train'
    val_jpg_folder = './data/gnt-jpg/validate'
    test_jpg_folder = './data/gnt-jpg/test'

    # Paths to train, validation, and test gnt folders
    train_gnt_folder = './data/gnt/train'
    val_gnt_folder = './data/gnt/validate'
    test_gnt_folder = './data/gnt/test'

    # Create label dictionary
    label_dict_train = create_label_dict(train_gnt_folder)
    label_dict_val = create_label_dict(val_gnt_folder)
    label_dict_test = create_label_dict(test_gnt_folder)

    # Create datasets and dataloaders
    train_dataset = ChineseDataset(train_jpg_folder, label_dict_train)
    val_dataset = ChineseDataset(val_jpg_folder, label_dict_val)
    test_dataset = ChineseDataset(test_jpg_folder, label_dict_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model
    model = NeuralNetwork().to('cuda')

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    # Training loop
    num_epochs = 60

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        i = 0
        for inputs, labels in train_loader:
            outputs = model(inputs.float().to('cuda'))
            loss = loss_fn(outputs, labels.squeeze().to('cuda'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print(f"Completed {i} batches")
            i += 1

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                if total == 3712:
                    break

                outputs = model(inputs.float().to('cuda'))
                loss = loss_fn(outputs, labels.squeeze().to('cuda'))
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += batch_size
                correct += predicted.eq(labels.to('cuda')).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        end_time = time.time()

        print(f"Epoch {epoch + 1} completed in {end_time - start_time} seconds")
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}')

    # Save the model
    torch.save(model.state_dict(), './models/hccr-model2.0.pth')

    # Test the model
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            if test_total == 14976:
                break
            outputs = model(inputs.float().to('cuda'))
            _, predicted = outputs.max(1)
            test_total += batch_size
            test_correct += predicted.eq(labels.to('cuda')).sum().item()

    test_accuracy = test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2%}') 
