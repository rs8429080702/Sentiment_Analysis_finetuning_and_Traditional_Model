from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
# Path to the fine-tuned model directory
# MODEL_DIR = "path_to_your_model_directory"

# Load model and tokenizer

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_path = "finetuned_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Function to prepare the input
def prepare_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

# Prediction function
def predict(text):
    model.eval()
    with torch.no_grad():
        inputs = prepare_text(text)
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class

# Streamlit App UI
st.title("Welcome to Mettler Toledo Sentiment Analysis Tool")
st.write("Enter a review or feedback to analyze its sentiment.")

review_text = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review_text.strip():
        prediction = predict(review_text)
        sentiment_mapping = {0: "Positive", 1: "Neutral", 2: "Negative"}
        sentiment = sentiment_mapping.get(prediction, "Unknown")
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text for analysis.")
