import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_utils import ensemble_predict

# Load dataset
df = pd.read_csv("final_dataset2.csv")
df["length"] = df["src_text"].str.split().str.len()

# Load models and tokenizers
afri_model = AutoModelForSequenceClassification.from_pretrained("afriberta_dir")
afri_tokenizer = AutoTokenizer.from_pretrained("afriberta_dir")
xlmr_model = AutoModelForSequenceClassification.from_pretrained("xlmr_dir")
xlmr_tokenizer = AutoTokenizer.from_pretrained("xlmr_dir")

afri_model.eval()
xlmr_model.eval()

# Overall evaluation
texts = df["src_text"].tolist()
labels = df["label"].tolist()
preds = []

for text in tqdm(texts, desc="Evaluating"):
    probs = ensemble_predict(text, afri_model, afri_tokenizer, xlmr_model, xlmr_tokenizer)
    pred_label = 1 if probs[1] > 0.5 else 0
    preds.append(pred_label)

print("\n🔹 Overall Metrics")
print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
print(f"F1 Score:  {f1_score(labels, preds):.4f}")
print(f"Precision: {precision_score(labels, preds):.4f}")
print(f"Recall:    {recall_score(labels, preds):.4f}")

# Language-specific evaluation
print("\n🔸 Per-Language Evaluation")
for lang in df["language"].unique():
    lang_df = df[df["language"] == lang]
    lang_texts = lang_df["src_text"].tolist()
    lang_labels = lang_df["label"].tolist()
    lang_preds = []
    for text in lang_texts:
        probs = ensemble_predict(text, afri_model, afri_tokenizer, xlmr_model, xlmr_tokenizer)
        pred_label = 1 if probs[1] > 0.5 else 0
        lang_preds.append(pred_label)
    print(f"{lang}: Accuracy = {accuracy_score(lang_labels, lang_preds):.4f}, F1 = {f1_score(lang_labels, lang_preds):.4f}")

# Robustness by text length
print("\n🔸 Robustness by Text Length")

def evaluate_by_length(subset_df, label):
    subset_texts = subset_df["src_text"].tolist()
    subset_labels = subset_df["label"].tolist()
    subset_preds = []
    for text in subset_texts:
        probs = ensemble_predict(text, afri_model, afri_tokenizer, xlmr_model, xlmr_tokenizer)
        pred_label = 1 if probs[1] > 0.5 else 0
        subset_preds.append(pred_label)
    print(f"{label}: Accuracy = {accuracy_score(subset_labels, subset_preds):.4f}, F1 = {f1_score(subset_labels, subset_preds):.4f}")

evaluate_by_length(df[df["length"] <= 15], "Short (<=15 words)")
evaluate_by_length(df[(df["length"] > 15) & (df["length"] <= 30)], "Medium (16–30 words)")
evaluate_by_length(df[df["length"] > 30], "Long (>30 words)")
