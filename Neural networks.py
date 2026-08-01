import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# parameters and hyperparameters
TRAIN_RATIO = 0.8
L_LAYERS = 3
D_NEURONS = 128
ETA = 0.001
N_EPOCHS = 15
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset preparation and split

# transforms MNIST images to pytorch tensors
transform = transforms.ToTensor()

# download full MNIST dataset
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# split dataset into X% train and (100-X)% test
train_size = int(TRAIN_RATIO * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# neural network architecture and initialization
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes, layers, neurons):
        super(SimpleNN, self).__init__()
        net_layers = []

        # input layer to first hidden layer
        prev_dim = input_dim
        for _ in range(layers):
            net_layers.append(nn.Linear(prev_dim, neurons))
            net_layers.append(nn.Sigmoid()) # all hidden layers use sigmoid
            prev_dim = neurons

        # output layer
        net_layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*net_layers)

        # custom weight and bias initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # normal distribution weight initiliazation with proper dispersion
                nn.init.xavier_normal_(m.weight)
                # initialize biases with zeros
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # flatten image inputs from (B, 1, 28, 28) to (B, 784)
        x = x.view(x.size(0), -1)
        return self.model(x)

model = SimpleNN(input_dim=28*28, num_classes=10, layers=L_LAYERS, neurons=D_NEURONS).to(DEVICE)

# loss function and optimizer

#L2 norm loss (mean squared error). One-hot encoding targets are required for MSE with classification.
criterion = nn.MSELoss()

# ADAM optimizer with learning rate eta
optimizer = optim.Adam(model.parameters(), lr=ETA)

# training and evaluation loop
train_losses = []
test_losses = []

for epoch in range(N_EPOCHS):
    # training phase
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(DEVICE)

        # convert integer labels to one-hot vectors for L2/MSE loss
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, one_hot_labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)

    epoch_train_loss = running_train_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    # testing phase
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, one_hot_labels)
            running_test_loss += loss.item() * images.size(0)

    epoch_test_loss = running_test_loss / len(test_dataset)
    test_losses.append(epoch_test_loss)

    print(f"Epoch [{epoch+1}/{N_EPOCHS}] - Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f}")


# Plotting results
plt.figure(figsize=(8,5))
plt.plot(range(1, N_EPOCHS + 1), train_losses, label="Train Loss (L2)", marker="o")
plt.plot(range(1, N_EPOCHS + 1), test_losses, label="Test Loss (L2)", marker="s")
plt.xlabel("Training Epoch")
plt.ylabel("L2 Norm loss")
plt.title("MNIST Training & test loss vs Epochs")
plt.legend()
plt.grid(True)

# displays
plt.show()