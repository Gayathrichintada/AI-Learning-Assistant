import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved files
model = load_model("models/saved/bilstm_model.keras")

with open("models/saved/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/saved/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

MAX_LEN = 100


def predict_emotion(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(sequence, maxlen=MAX_LEN)

    prediction = model.predict(padded, verbose=0)

    index = np.argmax(prediction)

    label = encoder.inverse_transform([index])[0]

    confidence = float(np.max(prediction))

    return label, confidence