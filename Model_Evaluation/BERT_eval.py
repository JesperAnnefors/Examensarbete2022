from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# hämtar och läser output.csv, som innehåller datan från datasetet sweparaphrase
path = "resources/output.csv"

df = pd.read_csv(
    path,
    header=None,
    names=[
        "original_id",
        "source",
        "type",
        "sentence_swe1",
        "sentence_swe2",
        "score",
        "sentence1",
        "sentence2",
    ],
)

# val av Kungliga Bibliotekets BERT-modell
model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

sentences1 = df["sentence_swe1"].tolist()
sentences2 = df["sentence_swe2"].tolist()

# skapar vektorrepresentationer genom Kungliga Bibliotekets BERT-modell
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# likheten mellan meningsparens vektorer beräknas genom Cosine Similarity
cosine_scores = cosine_similarity(embeddings1, embeddings2)
sentence_pair_scores = cosine_scores.diagonal()

# korrelationen mellan cosine-similarity värdet samt de manuellt satta likhetsbetygen i Sweparaphrase
# bedöms genom Pearsons och Spearmans korrelationskoefficienter
df["model_score"] = sentence_pair_scores.tolist()
print("Spearman:")
print(df[["score", "model_score"]].corr(method="spearman"))
print("\nPearson:")
print(df[["score", "model_score"]].corr(method="pearson"))