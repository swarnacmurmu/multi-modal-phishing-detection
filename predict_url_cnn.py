import torch
import torch.nn as nn

MODEL_PATH = "models/url_cnn/url_cnn_model.pt"
MAX_LEN = 200

def encode_url(url: str):
    encoded = [ord(c) % 71 for c in url[:MAX_LEN]]
    encoded += [0] * (MAX_LEN - len(encoded))
    return encoded

class CharCNN(nn.Module):
    def __init__(self, vocab_size=71):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, 32)
        self.conv = nn.Conv1d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(64 * 98, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x

model = CharCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def predict_url_cnn(url: str) -> int:
    x = torch.tensor([encode_url(url)], dtype=torch.long)
    with torch.no_grad():
        output = model(x).item()
    return 1 if output >= 0.5 else 0