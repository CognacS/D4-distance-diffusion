import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

# Simple Neural Network
class SimpleNN(pl.LightningModule):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.layer_2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# Generate dummy data
def generate_data(num_samples=1000, input_size=10, output_size=1):
    X = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, output_size)
    return TensorDataset(X, y)

# Main function to setup data, model, and trainer
def main():
    # Parameters
    input_size = 10
    output_size = 1
    batch_size = 32
    num_epochs = 10

    # Data
    dataset = generate_data()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = SimpleNN(input_size, output_size)

    # Trainer with DDPStrategy for multi-GPU, multi-node setup
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu',
        devices=1,  # Use one GPU per machine
        num_nodes=2,  # Number of machines
        strategy='ddp',
    )

    # Training
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    print("---- Starting Program")
    main()