import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model


# -----------------------------
# Load model + tokenizer
# -----------------------------
model_path = "model"
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()
# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Text Prediction")
st.snow()
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        st.balloons()
        # Tokenize input
        inputs = tokenizer(
            user_input, return_tensors="pt", truncation=True, padding=True
        )

        # Get prediction (classification)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=1).item()

        st.subheader("Prediction:")
        labels = {0: "Fake", 1: "Real"}
        st.write(f"Class: {labels[pred_class]}")
