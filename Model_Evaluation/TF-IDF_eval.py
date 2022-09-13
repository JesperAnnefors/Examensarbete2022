import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import stanza
import string
import os

# förbehandling av texter som ansågs krävas för TF-IDF-modellen
class LemmaTokenizer(object):
  def __init__(self):
    self.nlp = stanza.Pipeline('sv', processors='tokenize, pos, lemma')
  def __call__(self, articles):
    doc = self.nlp(articles)    
    return list(word.lemma for sent in doc.sentences for word in sent.words if word.lemma not in string.punctuation)

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

# TF-IDF-modellen
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())

sentences1 = df["sentence_swe1"].tolist()
sentences2 = df["sentence_swe2"].tolist()

# skapa inbäddningar enligt TF-IDF-modellen
embeddings1 = vectorizer.fit_transform(sentences1)
embeddings2 = vectorizer.fit_transform(sentences1)

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