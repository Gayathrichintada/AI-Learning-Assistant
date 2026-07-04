import os
import pickle
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# -------------------------
# Load Data
# -------------------------

train = pd.read_csv("data/train_processed.csv")
test = pd.read_csv("data/test_processed.csv")

# -------------------------
# Tokenizer
# -------------------------

tokenizer = Tokenizer(num_words=10000)

tokenizer.fit_on_texts(train["text"])

X_train = tokenizer.texts_to_sequences(train["text"])
X_test = tokenizer.texts_to_sequences(test["text"])

max_len = 100

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# -------------------------
# Labels
# -------------------------

encoder = LabelEncoder()

y_train = encoder.fit_transform(train["label"])
y_test = encoder.transform(test["label"])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# -------------------------
# Model
# -------------------------

model = Sequential()

model.add(
    Embedding(
        input_dim=10000,
        output_dim=128
    )
)

model.add(
    Bidirectional(
        LSTM(64)
    )
)

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dense(
        6,
        activation="softmax"
    )
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Train
# -------------------------

history = model.fit(

    X_train,
    y_train,

    epochs=5,

    batch_size=64,

    validation_data=(X_test, y_test)

)

os.makedirs("models", exist_ok=True)

model.save("models/bilstm_model.keras")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\nModel Saved Successfully!")