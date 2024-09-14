import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import tqdm
import pickle
from matplotlib.animation import FuncAnimation
import random
import math
from generategrid import generate_random_grid, generate_random_grid_points, apply_gravity, GRID_SIZE, NUM_CLASSES

def save_dataset(dataset, save_path):
    with open(os.path.join(save_path, 'tetromino_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {os.path.join(save_path, 'tetromino_dataset.pkl')}")

def load_dataset(save_path):
    with open(os.path.join(save_path, 'tetromino_dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    return dataset
class TetrominoDataset(Dataset):
    def __init__(self, num_samples, max_tetrominos=20, tetromino_function=generate_random_grid):
        self.samples = []
        self.tetromino_function = tetromino_function
        for _ in range(num_samples):
            original = self.tetromino_function(max_tetrominos)
            transformed = apply_gravity(original)
            self.samples.append((original, transformed))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original, transformed = self.samples[idx]
        # Hilariously, I forgot to encode empty space - so I end up passing it to the model as [0, 0, 0, 0] instead of [1, 0, 0, 0]
        # However, the np.argmax returns 0 for [0, 0, 0, 0], which translates to an output class of [1, 0, 0, 0]
        # Meaning the model uses [0, 0, 0, 0] for black when taking input, but outputs [1, 0, 0, 0] for black
        # This worked out fine - and since the original tests took days to run, no reason to change it
        # The output of the model makes perfect sense so all is good
        # Just something to be aware of if you are playing with this yourself
        return (torch.FloatTensor(original.transpose(2, 0, 1)),
                torch.LongTensor(np.argmax(transformed, axis=2)))
    
class TetrominoMLP(nn.Module):
    def __init__(self):
        super(TetrominoMLP, self).__init__()
        input_size = GRID_SIZE * GRID_SIZE * NUM_CLASSES
        hidden_size = 1024  # You can adjust this
        output_size = GRID_SIZE * GRID_SIZE * NUM_CLASSES
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # Ensure input is contiguous and flatten it
        x = x.contiguous().view(x.size(0), -1)
        # Pass through MLP layers
        x = self.layers(x)
        # Reshape output to match ConvNet shape
        return x.view(-1, NUM_CLASSES, GRID_SIZE, GRID_SIZE)
HIDDEN_SIZE = 64
class TetrominoCNN(nn.Module):
    def __init__(self):
        super(TetrominoCNN, self).__init__()
        # (GRID_SIZE - 2) layers
        self.intro = nn.Conv2d(NUM_CLASSES, HIDDEN_SIZE, kernel_size=3, padding=1)
        self.intro_bn = nn.BatchNorm2d(HIDDEN_SIZE)
        self.convs = nn.ModuleList([
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding=1) for _ in range(GRID_SIZE - 2)
        ]
        )
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(HIDDEN_SIZE) for _ in range(GRID_SIZE - 2)
        ]
        )
        self.outro = nn.Conv2d(HIDDEN_SIZE, NUM_CLASSES, kernel_size=3, padding=1)

    def forward(self, x):
        """x = self.intro(x)
        for conv in self.convs:
            x = conv(x)
            x = torch.relu(x)
        x = self.outro(x)
        return x"""
        x = self.intro_bn(self.intro(x))
        for conv, bn in zip(self.convs, self.batch_norms):
            x = bn(conv(x))
            x = torch.relu(x)
        x = self.outro(x)
        return x
INTERMEDIATE_SIZE = 64
class TetrominoDARC(nn.Module):
    def __init__(self):
        super(TetrominoDARC, self).__init__()
        self.intro_conv = nn.Conv2d(NUM_CLASSES, INTERMEDIATE_SIZE, kernel_size=3, padding=1)
        self.intro_bn = nn.BatchNorm2d(INTERMEDIATE_SIZE)
        self.middle_conv = nn.Conv2d(INTERMEDIATE_SIZE, INTERMEDIATE_SIZE, kernel_size=3, padding=1)
        self.middle_bn = nn.BatchNorm2d(INTERMEDIATE_SIZE)
        self.final_conv1 = nn.Conv2d(INTERMEDIATE_SIZE, NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.intro_bn(self.intro_conv(x)))
        for _ in range(GRID_SIZE - 2):
            x = torch.relu(self.middle_bn(self.middle_conv(x)))
        
        return self.final_conv1(x)

    

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.numel()
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
def train_model(model, train_loader, test_loader, num_epochs, device, save_path, suffix):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loading_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in loading_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            losses.append(loss.item())
            loading_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        evaluate_model(model, test_loader, device)
        torch.save(model.state_dict(), os.path.join(save_path, f'tetromino_model_{epoch+1}_{suffix}.pth'))
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(save_path, f'tetromino_model_{num_epochs}_{suffix}.pth'))
    print(f"Model saved to {os.path.join(save_path, f'tetromino_model_{num_epochs}_{suffix}.pth')}")
    return losses

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
mode = 'eval'
model_type = 'darc'
model_constructor = {
    'mlp': TetrominoMLP,
    'cnn': TetrominoCNN,
    'darc': TetrominoDARC
}
device = torch.device("mps")
if mode == 'train':
    train_dataset = load_dataset('.')
    test_dataset = TetrominoDataset(10000, 0.2)

    model = model_constructor[model_type]()
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1_000
    print(f"Number of parameters: {num_params:.2f}K")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    losses = train_model(model, train_loader, test_loader, 3, device, './checkpoints', model_type)
    # Save losses
    with open(f'./checkpoints/losses_{model_type}.pkl', 'wb') as f:
        pickle.dump(losses, f)
elif mode == 'eval':
    # Load the losses
    with open(f'./checkpoints/losses_{model_type}.pkl', 'rb') as f:
        losses = pickle.load(f)
    # Divide each loss by the number of batches

    # Plot the losses
    # Smooth the losses using an exponential moving average
    beta = 0.99
    smoothed_loss = losses[0]
    smoothed_losses = [smoothed_loss]
    for loss in losses[1:]:
        smoothed_loss = beta * smoothed_loss + (1 - beta) * loss
        smoothed_losses.append(smoothed_loss)

    model = model_constructor[model_type]()
    model.load_state_dict(torch.load(f'./tetromino_model_perfect.pth'))
    model.to(device)
    # Evaluate the model across occupancies 0 to 1
    occupancies = np.linspace(0, 1, 101)
    accuracies = []
    for occupancy in occupancies:
        test_dataset = TetrominoDataset(10000, occupancy, generate_random_grid_points)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        accuracy = evaluate_model(model, test_loader, device)
        accuracies.append(accuracy)
    #plt.plot(occupancies, accuracies)
    with open(f'./checkpoints/accuracies_{model_type}_perfect.pkl', 'wb') as f:
        pickle.dump(accuracies, f)
