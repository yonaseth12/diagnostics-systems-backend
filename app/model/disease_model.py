# model loading, prediction logic will go here 
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pickle
import re
from sklearn.preprocessing import StandardScaler
from typing import Dict

# Load tokenizer, BERT model, label encoder, scalers, and trained model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bert_model.eval()

# ----------- Projectors ----------- #
# These will be shared and defined outside the function when building the full model
bert_projector = nn.Linear(768, 768)       # Reduce BERT to 768
structured_projector = nn.Linear(5, 64)    # Project 5 structured fields to 64


# ---------- TEXT PREPROCESSING MODULE ---------- #
def clean_doctors_note(text: str) -> str:
    """
    Clean doctor's notes while preserving punctuation that carries clinical meaning.
    """
    if pd.isna(text):
        return ""
    # Lowercase
    text = text.lower()
    # Removing unwanted characters but preserve useful ones
    # We preserve / - () [] {} : and .
    allowed = r"[^a-z0-9\s\-\.\,\:\(\)\[\]\{\}/]"
    text = re.sub(allowed, "", text)
    # Removing extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# --- Production-time scaling function ---
def scale_structured_data_inference(X_new: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply the trained scaler to a new input row or batch.
    """
    X = X_new.copy()
    # Encode gender
    X['gender'] = X['gender'].map({'male': 0, 'female': 1})
    # Define numeric columns
    numeric_cols = ['age', 'body_temperature', 'systolic_bp', 'diastolic_bp']
    # Apply previously fitted scaler
    X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X


def get_mean_pooled_embeddings(tokenized_inputs: Dict[str, torch.Tensor], model: AutoModel, device="cpu") -> torch.Tensor:
    """
    Compute mean pooled embeddings from tokenized batch input using Bio_ClinicalBERT.
    """
    model.eval()
    model.to(device)

    input_ids = tokenized_inputs['input_ids'].to(device)
    attention_mask = tokenized_inputs['attention_mask'].to(device)
    token_type_ids = tokenized_inputs.get('token_type_ids', None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    # Last hidden state: [batch_size, seq_len, hidden_size]
    last_hidden_state = outputs.last_hidden_state

    # Expand attention mask to match hidden state
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    masked_hidden_state = last_hidden_state * mask

    # Mean pooling (summing non-masked and dividing by valid token count)
    summed = masked_hidden_state.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    mean_pooled = summed / counts

    return mean_pooled.cpu()  # Return to CPU for further processing 


def get_embedding_for_single_note(note: str, tokenizer, model: AutoModel, max_length=128, device="cpu") -> torch.Tensor:
    """
    Generate mean pooled embedding for a single doctor's note (used during inference).
    """
    tokenized = tokenizer(
        note,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=True
    )

    return get_mean_pooled_embeddings(tokenized, model, device=device)


# ----------- COMBINE MODULE (Production) ----------- #
def combine_features_inference_projected(X_structured_row: pd.DataFrame, bert_embedding: torch.Tensor) -> torch.Tensor:
    """
    Projects and combines a single structured input with BERT embedding (inference).
    Inputs:
      - X_structured_row: DataFrame of shape [1, 5]
      - bert_embedding: torch.Tensor [1, 768]
    Returns:
      - combined_features: [1, 832]
    """
    structured_tensor = torch.tensor(X_structured_row.values, dtype=torch.float32)
    proj_bert = bert_projector(bert_embedding)              # [1, 768]
    proj_struct = structured_projector(structured_tensor)   # [1, 64]
    combined = torch.cat([proj_bert, proj_struct], dim=1)   # [1, 832]
    return combined