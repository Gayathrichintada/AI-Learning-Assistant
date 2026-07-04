from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

dataset = load_dataset("emotion")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Dataset downloaded successfully!")
print(train_df.head())