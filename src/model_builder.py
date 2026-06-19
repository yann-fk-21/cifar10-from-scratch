"""
  Contains Pytorch model code to instantiate a TinyVGG
"""

import torch
from torch import nn

class TinyVGG(nn.Module):
  """
  Creates the TinyVGG architecture.

  Args:
    input_shape: The number of input channels.
    hidden_units: The number of hidden units.
    output_shape: The number of output classes.
  """
  def __init__(self, input_shape, hidden_units, output_shape):
    super().__init__()

    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(
            in_channels=input_shape,
            out_channels=hidden_units,
            kernel_size=3,
            padding=1,
            stride=1
        ),
        nn.Conv2d(
            in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            padding=1,
            stride=1
        ),
        nn.MaxPool2d(
            kernel_size=3
        )
    )

    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(
            in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            padding=1,
            stride=1
        ),
        nn.Conv2d(
            in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            padding=1,
            stride=1
        ),
        nn.MaxPool2d(
            kernel_size=3
        )
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units * 3 * 3, out_features=output_shape)
    )

  def forward(self, x) -> torch.Tensor:
    """
    Performs a forward pass through the model.

    Args:
      x: The input tensor.

    Returns:
      The output tensor from the classifier.
    """
    x = self.conv_block_1(x)
    x = self.conv_block_2(x)
    x = self.classifier(x)

    return x