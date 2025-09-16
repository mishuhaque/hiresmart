import pandas as pd
import random

# Example features
features = [
    "communication", "technical_skills", "quick_learner", "detail_oriented",
    "task_prioritization", "willing_to_learn", "teamwork", "leadership",
    "problem_solving", "adaptability", "creativity", "critical_thinking"
]

def random_profile():
    pos_traits = [
        "strong", "excellent", "good", "reliable", "outstanding"
    ]
    neg_traits = [
        "weak", "poor", "bad", "inconsistent", "lacking"
    ]
    profile = []
    for f in random.sample(features, k=5):
        if random.random() > 0.5:
            profile.append(f"{random.choice(pos_traits)} {f.replace('_',' ')}")
        else:
            profile.append(f"{random.choice(neg_traits)} {f.replace('_',' ')}")
    return ", ".join(profile)

# Generate dataset
examples = []
for i in range(500):
    profile = random_profile()
    label = "HIRE" if "strong" in profile or "excellent" in profile else "NOT HIRE"
    examples.append({
        "candidate_id": f"TRAIN_{i+1:03d}",
        "profile": f"Candidate Profile: {profile}",
        "label": label
    })

df = pd.DataFrame(examples)
df.to_csv("data/training_text_candidates.csv", index=False)
print("✅ Saved training dataset to data/training_text_candidates.csv")
