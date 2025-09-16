from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pandas as pd

# Load training data
train_df = pd.read_csv("data/training_text_candidates.csv")

# Convert training profiles to Documents
docs = [
    Document(page_content=row["profile"], metadata={"label": row["label"], "id": row["candidate_id"]})
    for _, row in train_df.iterrows()
]

# Build FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# Save index
db.save_local("data/candidate_index")

print("✅ Candidate RAG index built and saved")
