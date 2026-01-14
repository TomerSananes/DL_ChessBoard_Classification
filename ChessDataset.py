import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import PrePostProcessing as ppp


def upload_data_from_pc (root_dir):
    dataset_path_tags = []
    # Iterate through all game directories
    for game_dir in os.listdir(root_dir):
        game_path = os.path.join(root_dir, game_dir)
        if not os.path.isdir(game_path):
            continue

        # Locate the CSV file within the game directory
        csv_files = [f for f in os.listdir(game_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"Warning: No CSV found in {game_path}")
            continue

        csv_path = os.path.join(game_path, csv_files[0])
        # Load CSV (Assuming order: from_frame, to_frame, fen)
        df = pd.read_csv(csv_path)

        # Define path to the images folder
        images_dir = os.path.join(game_path, 'tagged_images')
        if not os.path.exists(images_dir):
            print(f"Warning: tagged_images folder missing in {game_path}")
            continue

        # Iterate through CSV rows to map frames to labels
        for _, row in df.iterrows():
            # row[0]=start, row[1]=end, row[2]=fen
            start_frame = int(row.iloc[0])
            end_frame = int(row.iloc[1])
            fen = row.iloc[2]

            # Convert FEN string to a numerical matrix
            matrix = ppp.fen_to_matrix(fen)

            # Map each frame in the range to its corresponding matrix label
            for frame_id in range(start_frame, end_frame + 1):
                img_name = f"frame_{frame_id:06d}.jpg"
                img_path = os.path.join(images_dir, img_name)

                if os.path.exists(img_path):
                    dataset_path_tags.append((img_path, matrix))

    return dataset_path_tags


class ChessDataset(Dataset):
    def __init__(self, samples, transform, is_train):
        # samples_list: List of tuples [(img_path, matrix_label), ...]
        # transform: PyTorch transformations (Resize, Normalize, etc.)
        self.transform = transform
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Extract path and label from the pre-collected list
        img_path, matrix = self.samples[idx]

        # Load image using PIL
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Apply random 90° rotation (0/90/180/270) during training
        if self.is_train:
            k = torch.randint(0, 4, (1,)).item()
            # rotates image counter-clockwise by k*90 degrees
            image = torch.rot90(image, k=k, dims=[1, 2])
            # rotates matrix counter-clockwise by k*90 degrees
            matrix = np.rot90(matrix, k=k).copy()

        # Return the image and the matrix as a LongTensor for CrossEntropyLoss
        return image, torch.tensor(matrix, dtype=torch.long)