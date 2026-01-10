import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from ChessDataset import ChessDataset
from Model import ChessModel
from tqdm import tqdm


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


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initial data loading without normalization
    base_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    full_dataset = ChessDataset(root_dir='./data/train', transform=base_transform)

    # Split dataset into 70% Train, 15% Validation, 15% Test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_indices, val_indices, test_indices = random_split(full_dataset, [train_size, val_size, test_size])

    # Compute normalization stats based only on the Train set to avoid data leakage
    train_subset = torch.utils.data.Subset(full_dataset, train_indices.indices)
    mean, std = get_mean_std(train_subset)
    print(f"Computed Stats - Mean: {mean}, Std: {std}")

    # Define final transformation including normalization
    final_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    # Apply the final transformation to the entire dataset
    full_dataset.transform = final_transform

    # Create DataLoaders for each split
    train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_indices.indices), batch_size=16, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_indices.indices), batch_size=16, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(full_dataset, test_indices.indices), batch_size=16, shuffle=False)

    # Initialize Model, Optimizer, and Loss Function
    model = ChessModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    all_train_loss = []
    all_val_loss = []
    epochs = 50
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
    torch.save(model.state_dict(), "chess_model_final.pth")


if __name__ == "__main__":
    train_model()