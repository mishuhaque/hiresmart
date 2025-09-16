import pandas as pd

# Load results
df = pd.read_csv("data/evaluation_results.csv")

# For now, keep only candidates with decision=HIRE
hire_df = df[df["decision"] == "HIRE"].copy()

# Add a mock probability score (based on explanation length as proxy)
hire_df["score"] = hire_df["explanation"].str.len()

# Sort and select top-5
top5 = hire_df.sort_values("score", ascending=False).head(5)

print("🎯 Top 5 Candidates Recommended for Hire:")
print(top5[["candidate_id", "decision", "score"]])

# Save to CSV
top5.to_csv("data/top5_candidates.csv", index=False)
print("✅ Top-5 saved to data/top5_candidates.csv")
