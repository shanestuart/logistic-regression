import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load dataset
heartdis = pd.read_csv("heart.csv")

X = heartdis[['cp','thal','ca','thalach','oldpeak']]
y = heartdis['target']

model = LogisticRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
