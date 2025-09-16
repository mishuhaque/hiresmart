import pandas as pd

# Load dataset
df = pd.read_csv("data/interview_candidates.csv")

# Define weights (customizable)
weights = {
    "communication": 0.2,
    "technical_skills": 0.3,
    "problem_solving": 0.25,
    "quick_learner": 0.15,
    "detail_oriented": 0.1
}

# Compute score
df["score"] = sum(df[col] * w for col, w in weights.items())

# Sort and show top 5
df = df.sort_values("score", ascending=False)
print("Top 5 Candidates (Rule-Based Ranking):")
print(df[["candidate_id", "score"]].head(5))
