import torch
import argparse
from torch import nn, optim
from torchvision import transforms
from data_setup import create_dataloader, get_preprocess_data
from engine import train
from model_builder import TinyVGG
from utils import save_model

def run_training(args):
    get_preprocess_data()

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dataloader, val_dataloader, class_names = create_dataloader(
        train_dir="data/cifar10/train",
        val_dir="data/cifar10/val",
        transforms=data_transform,
        batch_size=args.batch_size
    )
    

    model = TinyVGG(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(class_names)
    ).to(args.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=args.device
    )
    
    save_model(model=model, target_dir="models", model_name="tinyvgg_cifar10.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyVGG on CIFAR-10")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=10, help="Number of hidden units in TinyVGG")
    
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_training(args)
