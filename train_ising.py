import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# load data
data = torch.load('ising_data.pt')
X, y, T = data['X'], data['y'], data['T']

# dataset and DataLoader
dataset = TensorDataset(X, y, T)
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Simple CNN classifier
class IsingCNN(nn.Module):
    def __init__(self):
        super(IsingCNN, self).__init__()

        # layer 1: first RG Coarse-Graining Step
        # using circular padding respects periodic boundary conditions of the Ising lattice
        self.rg_block1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1, padding_mode='circular'), nn.ReLU(), nn.AvgPool2d(kernel_size=2, stride=2))

        # layer 2: second RG step
        self.rg_block2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode='circular'), nn.ReLU(), nn.AvgPool2d(kernel_size=2, stride=2))

        # global order parameter aggregation
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32*8*8, 64), nn.ReLU(), nn.Linear(64,2))

    def forward(self, x):
        x = self.rg_block1(x)
        x = self.rg_block2(x)
        return self.classifier(x)
    
model = IsingCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for imgs, labels, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] Loss : {total_loss/len(train_loader):.4f}")

print("training finished. Preparing plot...")

# analysis
model.eval()
temp_probs = {}
with torch.no_grad():
    for imgs, _, temps in test_loader:
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        for temp, prob in zip(temps.numpy(), probs.numpy()):
            temp_probs.setdefault(temp, []).append(prob)

mean_temps = sorted(temp_probs.keys())
mean_p_paramagnetic = [np.mean(temp_probs[t]) for t in mean_temps]

# plotting
plt.figure(figsize=(8,5))
plt.plot(mean_temps, mean_p_paramagnetic, 'o-', label='Predicted P(Paramagnetic)')
plt.axvline(x=2.269, color='r', linestyle='--', label='Theoretical $T_c \\approx 2.269$')
plt.xlabel('Temperature (T)')
plt.ylabel('Phase Probability')
plt.title('Neural network discovery of phase transition in 2D ising model')
plt.legend()
plt.grid(True)
plt.show()