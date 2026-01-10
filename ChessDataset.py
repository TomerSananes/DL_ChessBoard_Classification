import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import PrePostProcessing as ppp  # Your custom utility file


class ChessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

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
                        self.samples.append((img_path, matrix))

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Generates one sample of data given an index"""
        img_path, matrix = self.samples[idx]

        # Load image using PIL
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return the image and the matrix as a LongTensor for CrossEntropyLoss
        return image, torch.tensor(matrix, dtype=torch.long)