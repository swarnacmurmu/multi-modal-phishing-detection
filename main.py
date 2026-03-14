from src.inference.predict_email import predict_email
from src.inference.predict_url import predict_url
from src.inference.predict_url_cnn import predict_url_cnn
from src.fusion.decision_fusion import combine

def extract_basic_url_features(url: str) -> dict:
    domain = ""
    if "://" in url:
        parts = url.split("/")
        if len(parts) > 2:
            domain = parts[2]

    return {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsHTTPS": 1 if url.startswith("https") else 0
    }

if __name__ == "__main__":
    email_text = input("Enter email text: ")
    url = input("Enter URL: ")

    email_pred = predict_email(email_text)

    url_features = extract_basic_url_features(url)
    rf_pred = predict_url(url_features)

    cnn_pred = predict_url_cnn(url)

    final_result = combine(email_pred, rf_pred, cnn_pred)

    print("Email Prediction:", email_pred)
    print("Random Forest URL Prediction:", rf_pred)
    print("CNN URL Prediction:", cnn_pred)
    print("Final Result:", final_result)
