import torch
import torch.optim as optim
import torch.nn as nn
from data_utils import load_dataset
from model_utils import build_model

def train_model(data_folder, save_path="colorize.pth", epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = load_dataset(data_folder)
    model = build_model().to(device)
    opt = optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        for i, (L, ab) in enumerate(loader):
            L, ab = L.to(device), ab.to(device)
            out = model(L)
            loss = loss_fn(out, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print(f"Epoch {ep+1} Step {i} Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print("Model saved:", save_path)
