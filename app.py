import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('df.csv')  # Replace with your dataset path

# Load the trained model
model = load_model('lstm_threat_model.h5')

# Text preprocessing function (same as used during training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit app
st.title("Dataset Visualizations and Threat Score Prediction")

# Display the first few rows of the dataset
st.header("Dataset Overview")
st.write(df.head())

# Sidebar for selecting visualization
st.sidebar.header("Select Visualization")
visualization_type = st.sidebar.selectbox(
    "Choose a visualization type",
    ["Basic Statistics", "Histograms", "Scatter Plots", "Correlation Matrix", "Pairplot", "Box Plots", "Bar Plots"]
)

# Display the selected visualization
if visualization_type == "Basic Statistics":
    st.header("Basic Statistics")
    st.write(df.describe())

elif visualization_type == "Histograms":
    st.header("Histograms")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_column = st.sidebar.selectbox("Select Column", numerical_columns)
    st.subheader(f"Histogram for {selected_column}")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

elif visualization_type == "Scatter Plots":
    st.header("Scatter Plots")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_column = st.sidebar.selectbox("Select Column", numerical_columns)
    if selected_column != 'threat_score_normalized':  # Avoid plotting threat_score_normalized against itself
        st.subheader(f"Scatter Plot: threat_score_normalized vs {selected_column}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[selected_column], y=df['threat_score_normalized'], ax=ax)
        st.pyplot(fig)

elif visualization_type == "Correlation Matrix":
    st.header("Correlation Matrix")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    fig, ax = plt.subplots()
    sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif visualization_type == "Pairplot":
    st.header("Pairplot")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_columns = st.sidebar.multiselect("Select Columns", numerical_columns, default=numerical_columns)
    if selected_columns:
        fig = sns.pairplot(df[selected_columns])
        st.pyplot(fig)

elif visualization_type == "Box Plots":
    st.header("Box Plots")
    categorical_columns = df.select_dtypes(include=['object']).columns
    selected_column = st.sidebar.selectbox("Select Column", categorical_columns)
    st.subheader(f"Box Plot: threat_score_normalized vs {selected_column}")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[selected_column], y=df['threat_score_normalized'], ax=ax)
    st.pyplot(fig)

elif visualization_type == "Bar Plots":
    st.header("Bar Plots")
    categorical_columns = df.select_dtypes(include=['object']).columns
    selected_column = st.sidebar.selectbox("Select Column", categorical_columns)
    st.subheader(f"Bar Plot: {selected_column}")
    fig, ax = plt.subplots()
    sns.countplot(x=df[selected_column], ax=ax)
    st.pyplot(fig)

# Sidebar for threat score prediction
st.sidebar.header("Predict Threat Score")
new_report = st.sidebar.text_area("Report Text", "This is a new report text that needs to be preprocessed.")
new_complexity_score = st.sidebar.slider("Complexity Score", 0.0, 1.0, 0.7)
new_prevalence_score = st.sidebar.slider("Prevalence Score", 0.0, 1.0, 0.5)

if st.sidebar.button("Predict"):
    # Preprocess the new report text
    new_report = preprocess_text(new_report)

    # Tokenize and pad the new report text
    tokenizer = Tokenizer(num_words=5000)
    # Assuming the tokenizer was fitted on the training data
    tokenizer.fit_on_texts(df['report'])  # Use the same tokenizer as during training
    new_sequence = tokenizer.texts_to_sequences([new_report])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=100)

    # Normalize the new input scores
    scaler = MinMaxScaler()
    # Assuming the scaler was fitted on the training data
    scaler.fit(df[['complexity_score', 'prevalence_score']])  # Use the same scaler as during training
    new_scores = scaler.transform([[new_complexity_score, new_prevalence_score]])

    # Prepare the input data for the model
    X_text_new = new_padded_sequence
    X_scores_new = new_scores

    # Make the prediction
    predicted_threat_score = model.predict([X_text_new, X_scores_new])

    st.sidebar.write("Predicted Threat Score:", predicted_threat_score[0][0])