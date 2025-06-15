import pandas as pd
import numpy as np
import pickle
import torch
import os
from model.disease_model import tokenizer, bert_model, bert_projector, structured_projector, clean_doctors_note, scale_structured_data_inference, get_embedding_for_single_note, combine_features_inference_projected
from model.nn_disease_classifier import DiseaseClassifierSoftmax

# Common asset path
DM_ASSET_PATH = 'assets/city_dm_assets'

# Loading label encoder, scaler, and model from assets
with open(os.path.join(DM_ASSET_PATH, 'label_encoder_dm.pkl'), 'rb') as f:
    label_encoder_dm = pickle.load(f)
with open(os.path.join(DM_ASSET_PATH, 'scaler_dm.pkl'), "rb") as f:
    scaler_dm = pickle.load(f)

# Loading trained model and projectors
model = DiseaseClassifierSoftmax(num_classes=len(label_encoder_dm.classes_))
model.load_state_dict(torch.load(os.path.join(DM_ASSET_PATH, "disease_classifier_dm.pth")))
model.eval()

# Use the same projectors (must match what you used in training)
bert_projector.eval()
structured_projector.eval()



def predict_disease_dm(note: str, age: float, gender: str, body_temp: float, systolic: float, diastolic: float):
    predictions = predict_disease_dm_func(note, age, gender, body_temp, systolic, diastolic)
    print("🔍 Predicted Diseases:")
    for disease, prob in predictions[:5]:
        print(f"{disease}: {prob:.4f}")
    return predictions


def predict_disease_dm_func(note: str, age: float, gender: str, body_temp: float, systolic: float, diastolic: float):
    # Step 1: Create structured DataFrame
    X_struct = pd.DataFrame([{
        "age": age,
        "gender": gender.lower(),
        "body_temperature": body_temp,
        "systolic_bp": systolic,
        "diastolic_bp": diastolic
    }])

    # Step 2: Clean note
    cleaned_note = clean_doctors_note(note)

    # Step 3: Scale structured data
    X_struct_scaled = scale_structured_data_inference(X_struct, scaler_dm)

    # Step 4: Generate embedding for the note
    embedding = get_embedding_for_single_note(cleaned_note, tokenizer, bert_model)

    # Step 5: Combine features
    combined = combine_features_inference_projected(X_struct_scaled, embedding)

    # Step 6: Run through model
    with torch.no_grad():
        logits = model(combined)
        probs = torch.softmax(logits, dim=1).numpy().flatten()

    # Step 7: Decode and sort predictions
    predictions = dict(zip(label_encoder_dm.classes_, probs))
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    return sorted_preds


# Example usage
# predictions = predict_disease_dm(
#     note="frequent urination, excessive thirst, and unexplained weight loss",
#     age=35,
#     gender="male",
#     body_temp=38.5,
#     systolic=110,
#     diastolic=70
# )