from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("data/candidate_index", embeddings, allow_dangerous_deserialization=True)

# Example new candidate
new_candidate = "Candidate Profile: good technical skills, weak communication, reliable problem solving, strong adaptability"

# Retrieve similar candidates
retriever = db.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke(new_candidate)

print("🎯 New Candidate Evaluation")
print("Profile:", new_candidate)
print("\nTop Similar Past Candidates:")
for r in results:
    print("-", r.page_content, "| Label:", r.metadata["label"])
