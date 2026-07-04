from models.bilstm_predict import predict_emotion

label, confidence = predict_emotion(
    "I don't understand recursion."
)

print(label)
print(confidence)