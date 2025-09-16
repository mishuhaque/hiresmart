import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Load FAISS index (built from training_text_candidates.csv)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("data/candidate_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Load new candidates
new_df = pd.read_csv("data/interview_candidates.csv")

# Summarizer LLM
summarizer = pipeline("text2text-generation", model="google/flan-t5-base")

def evaluate_candidate(profile_text):
    # Retrieve similar candidates
    results = retriever.invoke(f"Candidate Profile: {profile_text}")
    
    # Collect labels and examples
    labels = [r.metadata["label"] for r in results]
    examples = [r.page_content + " | Label: " + r.metadata["label"] for r in results]
    
    # Decide majority vote
    decision = "HIRE" if labels.count("HIRE") > labels.count("NOT HIRE") else "NOT HIRE"
    
    # Summarize reasoning using LLM
    context = " ".join(examples)
    explanation = summarizer(
        f"Given the new candidate profile: {profile_text}. "
        f"Here are similar past candidates: {context}. "
        f"Explain why this candidate should be {decision}.",
        max_length=100
    )[0]["generated_text"]
    
    return decision, examples, explanation

# Run evaluation for all candidates
results = []
for _, row in new_df.iterrows():
    profile_text = ", ".join([f"{col}: {row[col]}" for col in new_df.columns if col != "candidate_id"])
    decision, examples, explanation = evaluate_candidate(profile_text)
    results.append({
        "candidate_id": row["candidate_id"],
        "decision": decision,
        "examples": examples,
        "explanation": explanation
    })

# Save results
output_df = pd.DataFrame(results)
output_df.to_csv("data/evaluation_results.csv", index=False)
print("✅ Evaluation complete. Results saved to data/evaluation_results.csv")
