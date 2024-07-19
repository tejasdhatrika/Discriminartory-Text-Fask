import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load the model
model = tf.keras.models.load_model('discriminatory_text_model.keras')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

class_labels = ['asian', 'black', 'chinese', 'jewish', 'latino', 'lgbtq', 'mental_dis', 'mexican', 'middle_east', 'muslim', 'native_american', 'physical_dis', 'women']

def preprocess_text(text):
    # Tokenize and pad sequences as per your preprocessing steps
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    return padded_sequences

# Streamlit app
st.title("Discriminatory Text Detection")

st.write("Enter the text to classify:")

text_input = st.text_area("Text")

if st.button("Predict"):
    if text_input:
        preprocessed_text = preprocess_text(text_input)
        prediction = model.predict(preprocessed_text)
        class_index = np.argmax(prediction, axis=1)[0]
        sentiment = class_labels[class_index]
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter some text.")
