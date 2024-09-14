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
GRID_SIZE = 17
NUM_CLASSES = 4  # black, blue, green, red

def create_empty_grid():
    return np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.uint8)

def create_tetromino(shape):
    tetrominos = {
        'I': np.array([[1, 1, 1, 1]]),
        'O': np.array([[1, 1], [1, 1]]),
        'T': np.array([[0, 1, 0], [1, 1, 1]]),
        'S': np.array([[0, 1, 1], [1, 1, 0]]),
        'Z': np.array([[1, 1, 0], [0, 1, 1]]),
        'J': np.array([[1, 0, 0], [1, 1, 1]]),
        'L': np.array([[0, 0, 1], [1, 1, 1]])
    }
    # Transpose the tetromino to match the grid

    return tetrominos[shape]

def place_tetromino(grid, tetromino, position, color):
    x, y = position
    h, w = tetromino.shape
    color_index = {'black': 0, 'blue': 1, 'green': 2, 'red': 3}
    
    if x + h > GRID_SIZE or y + w > GRID_SIZE:
        return False
    
    if np.any(grid[x:x+h, y:y+w, :].sum(axis=2) * tetromino):
        return False
    
    grid[x:x+h, y:y+w, color_index[color]] = tetromino
    
    return True

def generate_random_grid(max_tetrominos=50):
    grid = create_empty_grid()
    shapes = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
    colors = ['blue', 'green', 'red']
    # weighted between 0 and 50
    tetrominos_to_place = math.floor(random.random() * max_tetrominos)
    for _ in range(tetrominos_to_place):
        shape = np.random.choice(shapes)
        color = np.random.choice(colors)
        tetromino = create_tetromino(shape)
        
        for _ in range(50):  # Try 50 times to place each tetromino
            x = np.random.randint(0, GRID_SIZE - tetromino.shape[0] + 1)
            y = np.random.randint(0, GRID_SIZE - tetromino.shape[1] + 1)
            if place_tetromino(grid, tetromino, (x, y), color):
                break
    return grid

def generate_random_grid_points(occupancy=0.5):
    grid = create_empty_grid()
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if random.random() < occupancy:
                grid[x, y, np.random.choice([1, 2, 3])] = 1
    
    return grid
def apply_gravity(grid):
    new_grid = create_empty_grid()
    for col in range(GRID_SIZE):
        drop_height = GRID_SIZE - 1
        for row in range(GRID_SIZE - 1, -1, -1):
            if np.any(grid[row, col]):
                new_grid[drop_height, col] = grid[row, col]
                drop_height -= 1
    return new_grid
def apply_gravity_with_cellular_automata(grid):
    for i in range(GRID_SIZE - 1):
        new_grid = create_empty_grid()
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if np.any(grid[row, col]) and row < GRID_SIZE - 1 and not np.any(grid[row + 1, col]):
                    new_grid[row, col, :] = 0
                elif row > 0 and not np.any(grid[row, col]) and np.any(grid[row - 1, col]):
                    new_grid[row, col, :] = grid[row - 1, col, :]
                else:
                    new_grid[row, col, :] = grid[row, col, :]
        grid = new_grid
    return grid

def display_grids(original_grids, transformed_grids):
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    occupancies = np.linspace(0.0, 1.0, 5)
    fig.suptitle("Tetromino Grids: Original vs Gravity Applied", fontsize=16)

    for i, (orig, trans) in enumerate(zip(original_grids, transformed_grids)):
        for j, (grid, title) in enumerate([(orig, f"Occupancy {occupancies[i]}"), (trans, f"Transformed")]):
            ax = axes[j, i]
            rgb_grid = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
            rgb_grid[grid[:,:,1] == 1] = [0, 0, 255]  # Blue
            rgb_grid[grid[:,:,2] == 1] = [0, 255, 0]  # Green
            rgb_grid[grid[:,:,3] == 1] = [255, 0, 0]  # Red
            
            ax.imshow(rgb_grid)
            ax.set_title(title)
            ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
            ax.tick_params(which="minor", size=0)
            ax.tick_params(which="major", size=0)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
