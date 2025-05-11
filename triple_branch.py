import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import Mymodel


# reproducibility
random_seed = 959
random.seed(random_seed) # for python
rng = np.random.default_rng(random_seed) # new random number generation for numpy
np.random.seed(random_seed) # old random number generation for numpy
torch.manual_seed(random_seed) # pytorch
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

c1_rgb = np.load('c1_rgb.npy')
c2_rgb = np.load('c2_rgb.npy')

c1_time = np.load('c1_time.npy')
c2_time = np.load('c2_time.npy')

X_rgb = np.concatenate((c1_rgb, c2_rgb), axis=0)
X_time = np.concatenate((c1_time, c2_time), axis=0)
y = np.concatenate((np.zeros(len(c1_rgb)), np.ones(len(c2_rgb))))

X = list(zip(X_rgb, X_time))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=random_seed)

class MultiModalDataset(Dataset):

    def __init__(self, time_data, image_data, labels, transform=None):
        """
        time_data: numpy array, shape (N, 118, 5000)
        image_data: numpy array, shape (N, 224, 224, 3)
        labels: numpy array, shape (N,)
        """
        self.time_data = torch.tensor(time_data, dtype=torch.float32)
        self.image_data = torch.tensor(image_data, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, 3, 224, 224)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        time_feat = self.time_data[idx]
        image_feat = self.image_data[idx]
        label = self.labels[idx]

        if self.transform:
            image_feat = self.transform(image_feat)

        return time_feat, image_feat, label


time_train = np.array([x[1] for x in X_train])
image_train = np.array([x[0] for x in X_train])
time_test = np.array([x[1] for x in X_test])
image_test = np.array([x[0] for x in X_test])
y_train = np.array(y_train)
y_test = np.array(y_test)

# Create Dataset and DataLoader
train_dataset = MultiModalDataset(time_train, image_train, y_train)
test_dataset = MultiModalDataset(time_test, image_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_eval_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)  # 新增评估用loader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mymodel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Training cycle
epochs = 30
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    #  Training phase
    model.train()
    running_loss = 0.0
    for times, images, labels in train_loader:
        times, images, labels = times.to(device), images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(times, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluate the training set
    model.eval()
    train_running_loss = 0.0
    all_train_preds = []
    all_train_labels = []
    with torch.no_grad():
        for times, images, labels in train_eval_loader:  # 使用评估用loader
            times, images, labels = times.to(device), images.to(device), labels.to(device).float().unsqueeze(1)

            outputs = model(times, images)
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy().flatten()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy().flatten().astype(int))

    avg_train_loss = train_running_loss / len(train_eval_loader)
    train_acc = accuracy_score(all_train_labels, all_train_preds)

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

# Final test
model.eval()
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for times, images, labels in test_loader:
        times, images, labels = times.to(device), images.to(device), labels.to(device).float().unsqueeze(1)

        outputs = model(times, images)

        preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy().flatten()
        all_test_preds.extend(preds)
        all_test_labels.extend(labels.cpu().numpy().flatten().astype(int))

test_acc = accuracy_score(all_test_labels, all_test_preds)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Visualization of Results
save_dir = './results'
os.makedirs(save_dir, exist_ok=True)

# Training loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'train_loss_curve.png'))
plt.show()
plt.close()

# Training accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'train_accuracy_curve.png'))
plt.show()
plt.close()