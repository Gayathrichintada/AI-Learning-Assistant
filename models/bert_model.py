from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-uncased-emotion",
    tokenizer="bhadresh-savani/bert-base-uncased-emotion"
)

def predict_emotion(text):

    prediction = classifier(text)[0]

    return prediction["label"], prediction["score"]