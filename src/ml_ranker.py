import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/interview_candidates.csv")

# Add a dummy "hired" label for training (random for now)
np.random.seed(42)
df["hired"] = np.random.randint(0, 2, size=len(df))

X = df.drop(columns=["candidate_id", "hired"])
y = df["hired"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict probabilities
df["hire_probability"] = model.predict_proba(X)[:, 1]

# Rank by probability
top5 = df.sort_values("hire_probability", ascending=False).head(5)

print("Top 5 Candidates (ML Logistic Regression):")
print(top5[["candidate_id", "hire_probability"]])
