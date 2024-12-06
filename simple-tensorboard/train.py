import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import SimpleConvNet
import os

def download_mnist():
    """Download MNIST dataset if it doesn't exist"""
    if not os.path.exists('data'):
        os.makedirs('data')
    
    print("Downloading MNIST training dataset...")
    datasets.MNIST('data', train=True, download=True)
    print("Downloading MNIST test dataset...")
    datasets.MNIST('data', train=False, download=True)
    print("Download completed!")

def train_model():
    # First ensure we have the dataset
    download_mnist()

    print("Preparing data loaders...")
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    print("Initializing model...")
    # Model setup
    model = SimpleConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TensorBoard setup
    writer = SummaryWriter('runs/mnist_experiment')
    print("TensorBoard initialized - you can run 'tensorboard --logdir=runs' in a separate terminal")

    # Training loop
    n_epochs = 5
    
    # Log model graph
    sample_input = torch.randn(1, 1, 28, 28)
    writer.add_graph(model, sample_input)

    print("\nStarting training...")
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Log training metrics
            if batch_idx % 100 == 0:
                # Calculate training accuracy for this batch
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = 100. * correct / len(data)
                
                print(f'Epoch {epoch+1}/{n_epochs} Batch {batch_idx}/{len(train_loader)} '
                      f'Loss: {loss.item():.3f} Accuracy: {accuracy:.1f}%')
                
                # Log to tensorboard
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', loss.item(), step)
                writer.add_scalar('Accuracy/train', accuracy, step)

                # Log model weights histograms
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Parameters/{name}', param, step)

        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)

        # Log test metrics
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        
        print(f'Epoch {epoch+1}/{n_epochs} Test Loss: {test_loss:.3f} Accuracy: {accuracy:.1f}%')

    writer.close()
    print("\nTraining completed!")
    print("You can view the training metrics in TensorBoard")

if __name__ == '__main__':
    train_model()