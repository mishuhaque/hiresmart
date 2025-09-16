import pandas as pd
import numpy as np
import lightgbm as lgb

# Load dataset
df = pd.read_csv("data/interview_candidates.csv")

# Add dummy "hired" labels (for demo)
np.random.seed(42)
df["hired"] = np.random.randint(0, 2, size=len(df))

X = df.drop(columns=["candidate_id", "hired"])
y = df["hired"]

# One group containing all candidates (since we rank them together)
group = [len(X)]

ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg")
ranker.fit(X, y, group=group)

df["rank_score"] = ranker.predict(X)

# Top 5 candidates
top5 = df.sort_values("rank_score", ascending=False).head(5)

print("Top 5 Candidates (Learning-to-Rank with LightGBM):")
print(top5[["candidate_id", "rank_score"]])
