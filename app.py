from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('discriminatory_text_model.keras')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

class_labels = ['asian', 'black', 'chinese', 'jewish', 'latino', 'lgbtq','mental_dis', 'mexican', 'middle_east', 'muslim','native_american', 'physical_dis', 'women']

def preprocess_text(text):
    # Tokenize and pad sequences as per your preprocessing steps
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    return padded_sequences

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    class_index = np.argmax(prediction, axis=1)[0]
    sentiment = class_labels[class_index]
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
