from pathlib import Path
from typing import Union, List
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from Model import ChessModel
from PrePostProcessing import matrix_to_fen, fen_to_image


def run_inference(
    inputs: Union[str, Path, List[Union[str, Path]]],
    save_dir: Union[str, Path],
    model_path: str = "chess_model_final.pth",
):
    """
    Args:
        inputs: Path to a single image or list of image paths.
        model_path: Path to the trained model weights.
        save_dir: Optional directory to save the plot images. If None, images are not saved.

    Returns:
        List of predicted FEN strings.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load model
    model = ChessModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Prepare input
    if isinstance(inputs, (str, Path)):
        inputs = [Path(inputs)]
    else:
        inputs = [Path(p) for p in inputs]

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    # Preprocessing
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.6954, 0.6658, 0.5881],
                    std=[0.1977, 0.2170, 0.1869])
    ])

    predicted_fens = []

    # Loop over images
    for img_path in inputs:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor).to(device)
            probs = torch.softmax(output, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
            threshold = 0.7
            predictions[confidences < threshold] = 0
            pred_matrix = predictions[0].cpu().numpy()
            predicted_fen = matrix_to_fen(pred_matrix)

        predicted_fens.append(predicted_fen)

        # Render chessboard image
        board_img = fen_to_image(predicted_fen)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        axes[1].imshow(board_img)
        axes[1].set_title("Model Prediction")
        axes[1].axis("off")
        fig.tight_layout()

        # Save figure only
        if save_dir is not None:
            game_name = img_path.parent.parent.name
            game_id = game_name.split('_')[0]
            out_path = save_dir / f"{game_id}_{img_path.stem}_prediction.png"
            fig.savefig(out_path, dpi=200)

        plt.close(fig)
    return predicted_fens


if __name__ == "__main__":
    fen_list = run_inference('data/train/game2_per_frame/tagged_images/frame_001216.jpg' ,save_dir="test_inference")
    print(fen_list[0])
