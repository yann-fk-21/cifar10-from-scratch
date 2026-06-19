"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from torch import nn

from tqdm.auto import tqdm
from timeit import default_timer as timer
from typing import Dict, List, Tuple

def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

  """
  Trains a PyTorch model for one epoch.

  Args:
    model: The PyTorch model to be trained.
    dataloader: The DataLoader providing training data.
    loss_fn: The loss function used for optimization.
    optimizer: The optimizer used for weight updates.
    device: The device to perform computations on (e.g., 'cuda' or 'cpu').

  Returns:
    A tuple containing the average training loss and accuracy for the epoch.
  """

  model = model.to(device)
  model.train()

  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    y_logit = model(X)
    loss = loss_fn(y_logit, y)
    train_loss += loss.item()
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
    train_acc += (torch.eq(y_pred, y).sum().item() / len(y_pred))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc


def test_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module,
              device: torch.device) -> Tuple[float, float]:

  """
  Evaluates a PyTorch model for one epoch.

  Args:
    model: The PyTorch model to be evaluated.
    dataloader: The DataLoader providing test data.
    loss_fn: The loss function used for evaluation.
    device: The device to perform computations on (e.g., 'cuda' or 'cpu').

  Returns:
    A tuple containing the average test loss and accuracy for the epoch.
  """

  model.to(device)
  test_loss, test_acc = 0, 0
  model.eval()

  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      y_logit = model(X)
      loss = loss_fn(y_logit, y)
      test_loss += loss.item()
      y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
      test_acc += (torch.eq(y_pred, y).sum().item() / len(y_pred))

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


def train(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, loss_fn: nn.Module=nn.CrossEntropyLoss(), epochs: int = 5, device: torch.device=torch.device("cpu")) -> Dict[str, List[float]]:

  """
  Trains and evaluates a PyTorch model over a specified number of epochs.

  Args:
    model: The PyTorch model to be trained and evaluated.
    train_dataloader: The DataLoader providing training data.
    val_dataloader: The DataLoader providing validation data.
    optimizer: The optimizer used for weight updates.
    loss_fn: The loss function used for optimization.
    epochs: The number of training epochs.
    device: The device to perform computations on (e.g., 'cuda' or 'cpu').

  Returns:
    A dictionary containing lists of training and validation loss and accuracy
    for each epoch.
  """

  results = {
      "train_loss": [],
      "test_loss": [],
      "train_acc": [],
      "test_acc": []
  }

  start_time = timer()

  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
    test_loss, test_acc = test_step(model, val_dataloader, loss_fn, device)

    results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
    results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
    results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
    results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    print(f"Epoch {epoch + 1}:")
    print(f"Train loss: {train_loss: .3f} | Test loss: {test_loss: .3f}")
    print(f"Train acc: {train_acc: .3f} | Test acc: {test_acc: .3f}\n")

  end_time = timer()
  total_training_time = end_time - start_time
  print(f"Training time: {total_training_time: .3f} seconds")

  return results