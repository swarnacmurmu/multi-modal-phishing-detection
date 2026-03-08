import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from datasets import Dataset

# Load dataset
df = pd.read_csv("data/processed/email_clean.csv")
df = df.sample(n=10000, random_state=42)

print("Dataset shape:", df.shape)

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text_combined"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True
)

val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding=True
)

# Create dataset class
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, train_labels)
val_dataset = EmailDataset(val_encodings, val_labels)

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Training settings
training_args = TrainingArguments(
    output_dir="./models/email_model",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    evaluation_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Save model
model.save_pretrained("models/email_model")
tokenizer.save_pretrained("models/email_model")

print("Model training complete!")