# multi-modal-phishing-detection using Machine Learning


## Overview
This project detects phishing attacks by analyzing:

- Email semantics using DistilBERT
- URL character patterns using Character-Level CNN
- URL structural features using Random Forest

The system combines predictions from all models to improve phishing detection accuracy.

## Features
- Email phishing detection
- URL phishing detection
- Multi-model decision fusion
- Scalable ML architecture

## Tech Stack
Python, PyTorch, Transformers, TensorFlow, Scikit-learn

## Project Structure
data/ – datasets  
models/ – trained models  
src/ – training and inference code  
app/ – phishing detection API  

## Future Work
- Browser extension
- Screenshot-based phishing detection
- QR phishing detection
