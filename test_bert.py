from transformers import pipeline

emotion_classifier = pipeline(
    task="text-classification",
    model="bhadresh-savani/bert-base-uncased-emotion",
    tokenizer="bhadresh-savani/bert-base-uncased-emotion"
)

def predict_emotion(text):
    result = emotion_classifier(text)
    return result[0]