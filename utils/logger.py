import pandas as pd
import os
from datetime import datetime

FILE = "data/history.csv"

def save_history(problem, bilstm, bert, final_emotion, response):

    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Problem": problem,
        "BiLSTM": bilstm,
        "BERT": bert,
        "Final Emotion": final_emotion,
        "Response": response
    }

    # Handle missing or empty file
    if not os.path.exists(FILE) or os.path.getsize(FILE) == 0:
        df = pd.DataFrame(columns=row.keys())
    else:
        df = pd.read_csv(FILE)

    df.loc[len(df)] = row

    df.to_csv(FILE, index=False)