import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, BertModel
import joblib
from sklearn.preprocessing import StandardScaler

# --- Load Sentiment Analysis Model ---
repo_id_sentiment = "Raj8404/sentiment-analysis-model"  # Replace with your Hugging Face model path
model_sentiment = AutoModelForSequenceClassification.from_pretrained(repo_id_sentiment)
tokenizer_sentiment = AutoTokenizer.from_pretrained(repo_id_sentiment)

# --- Load Customer Satisfaction Model ---
# Load trained classifier model and scaler for Rating Prediction
model_classifier = joblib.load('final_logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load BERT model for embeddings
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_bert.to(device)
model_bert.eval()

# --- Functions ---
def prepare_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

# Sentiment Prediction
def predict_sentiment(text):
    inputs = prepare_text(text, tokenizer_sentiment, model_sentiment)
    model_sentiment.eval()
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class

# Customer Satisfaction Prediction (Rating 1 to 5)
def generate_embeddings(text):
    """ Generate embeddings for input text using BERT """
    inputs = tokenizer_bert(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_bert(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def preprocess_input(text):
    """ Preprocess the input text and generate features as done in training """
    embeddings = generate_embeddings(text)
    word_count = np.array([len(text.split())]).reshape(-1, 1)
    features_scaled = scaler.transform(word_count)
    features = np.hstack((embeddings, features_scaled))
    return features

# --- Streamlit UI ---
st.title("Customer Review Analysis App")

# Sidebar: Select Task
task = st.sidebar.selectbox("Select Task", ["Sentiment Analysis", "Predict Satisfaction"])

# --- Sentiment Analysis Task ---
if task == "Sentiment Analysis":
    st.subheader("Sentiment Analysis (Positive, Neutral, Negative)")

    review_text = st.text_area("Enter review text:")
    
    if st.button("Predict Sentiment"):
        if review_text:
            prediction = predict_sentiment(review_text)
            sentiment = ["Positive", "Neutral", "Negative"]
            st.write(f"Sentiment: {sentiment[prediction]}")
        else:
            st.write("Please enter some text.")

# --- Rating Prediction Task ---
else:
    st.subheader("Customer Satisfaction Prediction (1 to 5)")

    user_input = st.text_area("Enter your review:")

    if st.button('Predict Satisfaction'):
        if user_input:
            processed_input = preprocess_input(user_input)
            prediction = model_classifier.predict(processed_input)
            st.write(f'Predicted Satisfaction Rating: {prediction[0]}')
        else:
            st.write("Please enter some text.")


# import streamlit as st
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch

# # Load the model and tokenizer from Hugging Face Hub
# repo_id = "Raj8404/sentiment-analysis-model"  # Replace with your Hugging Face model path

# model = AutoModelForSequenceClassification.from_pretrained(repo_id)
# tokenizer = AutoTokenizer.from_pretrained(repo_id)

# # Check for GPU, otherwise use CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # Streamlit app interface
# st.title("Sentiment Analysis App")

# # Sidebar to select sentiment analysis task (Rating Prediction or Sentiment)
# task = st.sidebar.selectbox("Select Task", ["Predict Rating", "Sentiment Analysis"])

# # Function to prepare text for the model
# def prepare_text(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     return inputs

# # Prediction logic for Sentiment Analysis
# def predict_sentiment(text):
#     inputs = prepare_text(text)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         probabilities = torch.nn.functional.softmax(logits, dim=-1)
#         predicted_class = torch.argmax(probabilities, dim=-1).item()
#     return predicted_class

# # Prediction logic for Rating Prediction (assuming a different model or fine-tuned output)
# def predict_rating(text):
#     inputs = prepare_text(text)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         predicted_rating = torch.argmax(logits, dim=-1).item()  # Assuming 0-5 scale
#     return predicted_rating

# # Main content based on selected task
# if task == "Sentiment Analysis":
#     review_text = st.text_area("Enter review text:")
#     if st.button("Predict Sentiment"):
#         if review_text:
#             prediction = predict_sentiment(review_text)
#             sentiment = ["Positive", "Neutral", "Negative"]
#             st.write(f"Sentiment: {sentiment[prediction]}")
#         else:
#             st.write("Please enter some text.")
# else:
#     review_text = st.text_area("Enter review text:")
#     if st.button("Predict Rating"):
#         if review_text:
#             prediction = predict_rating(review_text)
#             st.write(f"Predicted Rating: {prediction}")
#         else:
#             st.write("Please enter some text.")
