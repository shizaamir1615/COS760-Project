# model_utils.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define prediction utilities here
def predict_proba(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()
    return probs

def ensemble_predict(text, afri_model, afri_tokenizer, xlmr_model, xlmr_tokenizer):
    probs_afri = predict_proba(text, afri_model, afri_tokenizer)
    probs_xlmr = predict_proba(text, xlmr_model, xlmr_tokenizer)
    avg_probs = [(a + b) / 2 for a, b in zip(probs_afri, probs_xlmr)]
    return avg_probs
