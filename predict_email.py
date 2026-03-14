


import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = "models/email_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def predict_email(email_text):
    """
    Returns phishing probability for an email text.
    Assumes:
    class 0 = legitimate
    class 1 = phishing
    """
    if not isinstance(email_text, str) or not email_text.strip():
        return 0.0

    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        phishing_score = probs[0][1].item()

    return float(phishing_score)


if __name__ == "__main__":
    sample_email = "Your account has been suspended. Click here immediately to verify."
    score = predict_email(sample_email)
    print("Email phishing score:", score)
