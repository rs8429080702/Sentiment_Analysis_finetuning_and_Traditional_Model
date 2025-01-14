# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import streamlit as st
# # Path to the fine-tuned model directory
# # MODEL_DIR = "path_to_your_model_directory"

# # Load model and tokenizer

# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # Load the model and tokenizer
# @st.cache_resource
# def load_model():
#     model_path = "finetuned_model"
#     model = AutoModelForSequenceClassification.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     return model, tokenizer, device

# model, tokenizer, device = load_model()

# # Function to prepare the input
# def prepare_text(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     return inputs

# # Prediction function
# def predict(text):
#     model.eval()
#     with torch.no_grad():
#         inputs = prepare_text(text)
#         outputs = model(**inputs)
#         logits = outputs.logits
#         probabilities = torch.nn.functional.softmax(logits, dim=-1)
#         predicted_class = torch.argmax(probabilities, dim=-1).item()
#     return predicted_class

# # Streamlit App UI
# st.title("Welcome to Mettler Toledo Sentiment Analysis Tool")
# st.write("Enter a review or feedback to analyze its sentiment.")

# review_text = st.text_area("Enter your review:")

# if st.button("Analyze"):
#     if review_text.strip():
#         prediction = predict(review_text)
#         sentiment_mapping = {0: "Positive", 1: "Neutral", 2: "Negative"}
#         sentiment = sentiment_mapping.get(prediction, "Unknown")
#         st.write(f"Predicted Sentiment: {sentiment}")
#     else:
#         st.warning("Please enter some text for analysis.")


import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the model and tokenizer from Hugging Face Hub
repo_id = "Raj8404/sentiment-analysis-model"  # Replace with your Hugging Face model path

model = AutoModelForSequenceClassification.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Check for GPU, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Streamlit app interface
st.title("Sentiment Analysis App")

# Sidebar to select sentiment analysis task (Rating Prediction or Sentiment)
task = st.sidebar.selectbox("Select Task", ["Predict Rating", "Sentiment Analysis"])

# Function to prepare text for the model
def prepare_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

# Prediction logic for Sentiment Analysis
def predict_sentiment(text):
    inputs = prepare_text(text)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class

# Prediction logic for Rating Prediction (assuming a different model or fine-tuned output)
def predict_rating(text):
    inputs = prepare_text(text)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_rating = torch.argmax(logits, dim=-1).item()  # Assuming 0-5 scale
    return predicted_rating

# Main content based on selected task
if task == "Sentiment Analysis":
    review_text = st.text_area("Enter review text:")
    if st.button("Predict Sentiment"):
        if review_text:
            prediction = predict_sentiment(review_text)
            sentiment = ["Positive", "Neutral", "Negative"]
            st.write(f"Sentiment: {sentiment[prediction]}")
        else:
            st.write("Please enter some text.")
else:
    review_text = st.text_area("Enter review text:")
    if st.button("Predict Rating"):
        if review_text:
            prediction = predict_rating(review_text)
            st.write(f"Predicted Rating: {prediction}")
        else:
            st.write("Please enter some text.")
