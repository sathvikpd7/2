# program 8

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_size, hidden_size, image_size = 64, 256, 784
batch_size, num_epochs, learning_rate = 100, 10, 0.0002

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = datasets.MNIST('data/', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, image_size), nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G, D = Generator().to(device), Discriminator().to(device)
criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)

# Display generated images
def show_images(images):
    grid = torchvision.utils.make_grid(
        images.reshape(-1, 1, 28, 28), nrow=10, normalize=True
    )
    plt.figure(figsize=(10, 4))
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis('off')
    plt.show()

# Training
for epoch in range(num_epochs):
    for images, _ in data_loader:
        n = images.size(0)
        images = images.reshape(n, -1).to(device)
        real_labels = torch.ones(n, 1, device=device)
        fake_labels = torch.zeros(n, 1, device=device)

        # Train discriminator
        real_loss = criterion(D(images), real_labels)
        z = torch.randn(n, latent_size, device=device)
        fake_images = G(z)
        fake_loss = criterion(D(fake_images.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train generator
        z = torch.randn(n, latent_size, device=device)
        fake_images = G(z)
        g_loss = criterion(D(fake_images), real_labels)

        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            show_images(G(torch.randn(100, latent_size, device=device)))

print("Training completed.")
