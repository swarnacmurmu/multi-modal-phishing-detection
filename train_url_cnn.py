import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os

# Load dataset
df = pd.read_csv("data/url_detection/url_dataset.csv")

urls = df["URL"].astype(str)
labels = df["label"]

# Character vocabulary
all_chars = sorted(list(set("".join(urls))))
char2idx = {c: i+1 for i, c in enumerate(all_chars)}

max_len = 200

def encode_url(url):
    encoded = [char2idx.get(c, 0) for c in url[:max_len]]
    encoded += [0] * (max_len - len(encoded))
    return encoded

X = [encode_url(url) for url in urls]
y = labels.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class URLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(URLDataset(X_train, y_train), batch_size=64, shuffle=True)

class CharCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, 32)
        self.conv = nn.Conv1d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(64 * 98, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x

model = CharCNN(len(char2idx))

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training URL Character CNN...")

for epoch in range(5):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/5 Loss: {total_loss:.4f}")

os.makedirs("models/url_cnn", exist_ok=True)

torch.save(model.state_dict(), "models/url_cnn/url_cnn_model.pt")

print("URL CNN model saved to models/url_cnn/")