import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():

    # Create EfficientNet-B4 architecture
    model = EfficientNet.from_name("efficientnet-b4")

    # Replace classifier with your trained 5-class classifier
    model._fc = nn.Sequential(
        nn.Linear(1792, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 5)
    )

    # Load trained weights
    checkpoint = torch.load(
        "best_dr_model.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint)

    # Move model to CPU/GPU
    model.to(device)

    # Evaluation mode
    model.eval()

    return model