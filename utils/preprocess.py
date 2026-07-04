import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()

    words = word_tokenize(text)

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(words)


def load_file(filename):

    texts = []
    labels = []

    with open(filename, encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line == "":
                continue

            text, label = line.rsplit(";", 1)

            texts.append(clean_text(text))
            labels.append(label)

    return pd.DataFrame({
        "text": texts,
        "label": labels
    })


train = load_file("data/train.txt")
test = load_file("data/test.txt")
val = load_file("data/val.txt")

train.to_csv("data/train_processed.csv", index=False)
test.to_csv("data/test_processed.csv", index=False)
val.to_csv("data/val_processed.csv", index=False)

print(train.head())

print("\nDataset processed successfully!")