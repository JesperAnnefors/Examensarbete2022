import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import stanza
import fasttext.util
import string
import os

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

# förbehandling av texter som ansågs krävas för FastText-modellen
def preProcess(text):
  doc = nlp(text)
  return " ".join(list(word.lemma for sent in doc.sentences for word in sent.words if word.lemma not in string.punctuation))

# hämtande av FastText-modell
fasttext.util.download_model('sv', if_exists='ignore')
model = fasttext.load_model('cc.sv.300.bin')

# pipeline för förbehandling av texter
nlp = stanza.Pipeline('sv', processors='tokenize, pos, lemma')

sentences1 = df["sentence_swe1"].tolist()
sentences2 = df["sentence_swe2"].tolist()

# förbehandling av texter som ansågs krävas för FastText-modellen
processed1 = []
for sentence in sentences1:
    processed1.append(preProcess(sentence))

processed2 = []
for sentence in sentences2:
    processed2.append(preProcess(sentence))

# skapar vektorrepresentationer genom FastText-modellen
embeddings1 = []
for text in processed1:
  embeddings1.append(model.get_sentence_vector(text))

embeddings2 = []
for text in processed2:
  embeddings2.append(model.get_sentence_vector(text))

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