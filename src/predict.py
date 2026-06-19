"""
Contains inference function for prediction
"""

import torch
import torch.nn as nn
from typing import List, Optional
from pathlib import Path
import torchvision

# Assuming device is defined or imported, here is a fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_prediction(model: nn.Module, image_path: str, transforms: Optional[object], class_names: List[str], device: torch.device = device):
    """
    Makes a prediction on a target image with a trained model and returns the prediction result.

    Args:
        model (nn.Module): A trained PyTorch model.
        image_path (str): Path to the target image.
        transforms (callable, optional): A series of transforms to apply to the image.
        class_names (List[str]): A list of class names to map the prediction index to a class name.
        device (torch.device, optional): The device to run the prediction on (defaults to CPU/GPU).

    Returns:
        tuple: (prediction_probability, class_name)
    """
    model = model.to(device)

    image_path = Path(image_path)
    readed_image = torchvision.io.read_image(str(image_path))
    readed_image = readed_image / 255
    readed_image = readed_image.to(device)

    if transforms:
        readed_image = transforms(readed_image)

    with torch.inference_mode():
        readed_image = torch.unsqueeze(readed_image, dim=0)
        y_logit = model(readed_image)
        y_pred_probs = torch.softmax(y_logit, dim=1)
        y_pred = torch.argmax(y_pred_probs, dim=1)
        y_pred_prob = y_pred_probs.cpu().max().item()
        class_pred = class_names[y_pred]

        return y_pred_prob, class_pred
