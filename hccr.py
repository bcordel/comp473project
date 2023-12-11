import io
import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import os
from PIL import Image

torch.cuda.set_per_process_memory_fraction(0.9)
torch.cuda.empty_cache()


class ChineseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.gnt')]
        self.samples = []
        self.labels = []
        self.num_classes = 0

        for file_name in self.image_files:
            samples, labels = self.gnt_read_images(file_name)
            self.samples.extend(samples)
            self.labels.extend(labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        label = self.labels[idx]

        # Apply transformations if any
        if self.transform:
            img = self.pad_to_size(self.transform(img), 256)
            label_int_sequence = [ord(char) for char in label.decode('gb18030')]
            label_mapped = [ord(char) % 3755 for char in label.decode('gb18030')]

            label_tensor = torch.tensor(label_mapped)

        sample = {'image': img, 'label': label_tensor}

        return sample

    def pad_to_size(self, tensor, target_size):
        _, h, w = tensor.size()

        # Calculate padding values
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)

        # Apply padding
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=255)

        return tensor

    def gnt_read_images(self, file_name):
        samples = []
        labels = []

        try:
            with open(file_name, 'rb') as image_file:
                image_file.seek(0, 2)
                file_length = image_file.tell()
                image_file.seek(0, 0)

                while image_file.tell() < file_length:
                    # skip length
                    int.from_bytes(image_file.read(4), byteorder='little')
                    # image label
                    label = image_file.read(2)
                    # image dimensions
                    width = int.from_bytes(image_file.read(2), byteorder='little')
                    height = int.from_bytes(image_file.read(2), byteorder='little')
                    # bitmap of gray-scale image
                    image_bitmap = bytearray(image_file.read(width * height))

                    img = Image.frombytes('L', (width, height), bytes(image_bitmap))

                    samples.append(img)
                    labels.append(label)

                self.num_classes = len(set(self.labels))

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        return samples, labels


# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 8192),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3755)
        )

    def forward(self, x):
        return self.model(x)


# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to(torch.device('cuda'))
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    chinese_dataset = ChineseDataset(root_dir="data/temp", transform=ToTensor())
    validation_dataset = ChineseDataset(root_dir="data/competition-gnt", transform=ToTensor())
    dataset = DataLoader(chinese_dataset, batch_size=4, shuffle=True)

    for epoch in range(2):
        batch_num = 0
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in dataset:
            print(f"Batch {batch_num} started")
            X, y = batch['image'], batch['label']
            X = X.float().to(torch.device('cuda'))
            y = y.squeeze(dim=1).to(torch.device('cuda'))
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Update accuracy
            _, predicted_labels = torch.max(yhat, 1)
            correct_predictions += (predicted_labels == y).sum().item()
            total_samples += y.size(0)

            total_loss += loss.item()
            batch_num += 1
            torch.cuda.empty_cache()

        # Calculate accuracy and print metrics after each epoch
        accuracy = correct_predictions / total_samples
        average_loss = total_loss / len(dataset)

        print(f"Epoch: {epoch}, Loss: {average_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

        # Validation set evaluation (assuming you have a validation dataset)
        clf.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            total_val_loss = 0.0
            val_correct_predictions = 0
            total_val_samples = 0

            for val_batch in validation_dataset:
                val_X, val_y = val_batch['image'].float().to(torch.device('cuda')), val_batch['label'].squeeze(dim=1).to(torch.device('cuda'))
                val_yhat = clf(val_X)
                val_loss = loss_fn(val_yhat, val_y)

                _, val_predicted_labels = torch.max(val_yhat, 1)
                val_correct_predictions += (val_predicted_labels == val_y).sum().item()
                total_val_samples += val_y.size(0)

                total_val_loss += val_loss.item()

            val_accuracy = val_correct_predictions / total_val_samples
            average_val_loss = total_val_loss / len(validation_dataset)

            print(f"Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

        clf.train()

    with open('model_state_chinese.pt', 'wb') as f:
        save(clf.state_dict(), f)

    with open('model_state_chinese.pt', 'rb') as f:
        clf.load_state_dict(load(f))