import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import ChessDataset
from Model import ChessModel
from tqdm import tqdm
import PrePostProcessing as ppp


def get_mean_std (train_dataset):
    """Calculates mean and standard deviation for the training set only."""
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    total_images = 0

    print("Calculating mean and std on train set...")
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        # Reshape images to (batch_size, channels, pixels)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean, std


def calculate_accuracy(outputs, labels):
    """Calculates the percentage of accurately predicted squares."""
    preds = torch.argmax(outputs, dim=-1)
    correct = (preds == labels).float().sum()
    total = labels.numel()
    return (correct / total).item()


def get_weights():
    counts = ppp.fast_check_distribution("./data/train")
    # Calculate inverse weights to handle data imbalance
    counts_list = [counts[i] for i in range(13)]
    counts_tensor = torch.tensor(counts_list, dtype=torch.float)
    weights = 1.0 / counts_tensor

    # Normalize weights so they sum up to the number of classes
    weights = weights / weights.sum() * 13

    for i, w in enumerate(weights):
        print(f"Class {i}: {w:.4f}")

    return weights


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set Seed for Reproducibility
    seed = 42
    generator = torch.Generator().manual_seed(seed)

    base_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    # load raw data
    raw_data = ChessDataset.upload_data_from_pc(root_dir='./data/train')

    # Split dataset into 70% Train, 15% Validation, 15% Test
    train_size = int(0.7 * len(raw_data))
    val_size = int(0.15 * len(raw_data))
    test_size = len(raw_data) - train_size - val_size
    train_list, val_list, test_list = random_split(raw_data, [train_size, val_size, test_size], generator=generator)

    train_dataset = ChessDataset.ChessDataset(list(train_list), transform=base_transform, is_train=True)
    val_dataset = ChessDataset.ChessDataset(list(val_list), transform=base_transform, is_train=False)
    test_dataset = ChessDataset.ChessDataset(list(test_list), transform=base_transform, is_train=False)


    # Compute normalization stats based only on the train set to avoid data leakage
    mean, std = get_mean_std(train_dataset)

    # Define final transformation including normalization
    final_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    # Define train final transformation including normalization
    final_train_transform = T.Compose([
        T.Resize((256, 256)),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    # Apply the final transformation to each dataset
    train_dataset.transform = final_train_transform
    val_dataset.transform = final_transform
    test_dataset.transform = final_transform

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Initialize Model, Optimizer, and Loss Function
    model = ChessModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    criterion = nn.CrossEntropyLoss(weight=get_weights().to(device))

    # Training Loop
    all_train_loss = []
    all_val_loss = []
    epochs = 200
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            # Reshape for CrossEntropy: (Batch*64, Classes) and (Batch*64)
            loss = criterion(outputs.view(-1, 13), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_train_loss.append(loss.item())
            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, labels)

        # Validation Phase
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs.view(-1, 13), labels.view(-1)).item()
                all_val_loss.append(loss)
                val_loss += loss
                val_acc += calculate_accuracy(outputs, labels)

        # Print epoch results
        print(f"Epoch {epoch + 1}:")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {val_acc / len(val_loader):.4f}")

    # Final Evaluation on the Test set
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            test_acc += calculate_accuracy(outputs, labels)

    print(f"\nFINAL TEST ACCURACY: {test_acc / len(test_loader):.4f}")

    # Save the trained weights
    torch.save(model.state_dict(), "chess_model.pth")


if __name__ == "__main__":
    train_model()