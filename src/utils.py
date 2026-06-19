"""
 Contains functions for save and load Pytorch model
"""


import torch
from torch import nn
from pathlib import Path


def save_model(model: nn.Module, target_dir: str, model_name: str):
  """
  Saves a PyTorch model to a target directory.

  Args:
    model: The PyTorch model to be saved.
    target_dir: The directory to save the model to.
    model_name: The filename for the saved model (must include .pth or .pt).
  """
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  assert model_name.endswith(".pth") or model_name.endswith(".pt")
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to {model_save_path}")
  torch.save(f=model_save_path, obj=model.state_dict())
    
    

def load_model(model: nn.Module, targ_dir: str):
  """
  Loads a PyTorch model from a target directory.

  Args:
    model: The PyTorch model to load the state dictionary into.
    targ_dir: The path to the saved model file.

  Returns:
    The model with the loaded state dictionary.
  """
  model_path = Path(targ_dir)
  return model.state_dict(torch.load(f=model_path))
    
    