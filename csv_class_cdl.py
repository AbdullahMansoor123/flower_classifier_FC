from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from def_cnn import CNN
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image


###Custom DataLoader###
class FlowerDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, data_dir, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): path of the csv file with annotation
            data_dir (string): Directory with all the images
            transform (callable, optional):preprocess transformation on the image
        """
        self.img_labels = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path) #reading image using skimage
        label = self.img_labels.iloc[idx, 1] #take labels from the class columns only

        if self.transform: #apply transformation to a single image
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    ##

transformer = transforms.Compose([
            transforms.Resize(255),
            transforms.RandomCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])


#Loading Data with transformation

transformed_data = FlowerDataset(csv_file='flower_images/flower_labels.csv',
                                 data_dir='flower_images/', transform=None,
                                 target_transform=None)
dataset_len = len(transformed_data)

# train and test split
train_len, test_len = dataset_len - 100, 100  # 1000 represent the size of testset we want
train_set, test_set = torch.utils.data.random_split(transformed_data, [train_len, test_len])


# train and test split
train_len = len(transformed_data)
# print(len(transformed_data))

# batch_size = 10
# train_dataloader = DataLoader(dataset=transformed_data, batch_size=batch_size, shuffle=True)
#
# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.title(f'the flower class is {label}')
# plt.axis('off')
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")



# hyper-parameters
num_classes = 2
batch_size = 10
learning_rate = 0.001
num_epochs = 5

# train and test dataloader
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
print(train_loader)


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('{} in use'.format(device))


# CNN Module
model = CNN(num_classes=num_classes)
model = model.to(device)
print(model)


# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    print(f'epoch:{epoch + 1}/{num_epochs}...........\n')
    total_correct = 0.0
    running_loss = 0.0
    total = 0
    for batch, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        total += labels.size(0)

        # Forward Pass
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        total_correct += (labels == predictions).sum().item()

        # backward
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        running_loss += loss.item() * images.size(0)
        loss.backward()
        # Optimizing
        optimizer.step()
    print(f'train loss: {(running_loss / total):.4f} train accuracy: {((total_correct / total) * 100):.2f}%')

# test loop
    with torch.no_grad():
        model.eval()  # notify our layer we are in evaluation mode
        total_loss, total_correct = 0.0, 0.0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (labels == predictions).sum().item()
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
    print(f'test loss: {(total_loss / total):.4f} test accuracy: {((total_correct / total) * 100):.2f}%\n')
print('Training and testing completed!')

## saving the model

