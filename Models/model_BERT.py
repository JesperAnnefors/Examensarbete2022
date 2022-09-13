import pandas as pd
import stanza
from sentence_transformers import SentenceTransformer
import numpy as np

class model_BERT:
  
  # konstruktor. pathToText är filvägen för patientberättelserna. amount är mängden patientberättelser
  def __init__(self, pathToText, amount):
    self.pathToText = pathToText
    self.amount = amount

  # skapar vektorrepresentationer
  def getVectors(self):

    # läser excelfilen med patientberättelserna. Patientberättelserna måste ligga under en kolumn som heter "Sammanfattning"
    data = pd.read_excel(self.pathToText, nrows=self.amount)
    stories = pd.DataFrame(data, columns=['Sammanfattning'])

    # förbehandlar text via Stanza
    nlp = stanza.Pipeline('sv', processors='tokenize')
    sentences_df = stories.apply(lambda x: pd.Series(self.__sentencizer(x,nlp)),axis=1)

    # hämtar Kungliga Bubliotekets BERT-modell
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

    # skapar vektorrepresentationer av patientberättelserna
    sentenceList = sentences_df["Sammanfattning"].tolist()
    
    embeddings = []
    for list in sentenceList:
      embeddings.append(model.encode(list))
    
    # BERT skapar vektorrepresentationer för hela meningar. 
    # För att representera texter på mer än en mening så tar man ett medelvärde på textens alla BERT-vektorer.
    vectors = []
    for embedding in embeddings:
      vectors.append(self.__averageVector(embedding))

    return vectors

# delar upp en text på flera meningar till en lista av dess meningar
  def __sentencizer (self, x, nlp_model):
    sentences = {"Sammanfattning":[]}
    doc = nlp_model(x["Sammanfattning"])
    for sent in doc.sentences:
      sentences["Sammanfattning"].append(sent.text.replace('.',''))
    return sentences


  # tar ett medelvärde på ett antal vektorer
  def __averageVector (self, vectors):
    sum = np.array([0.0] * len(vectors[0]))
    for vector in vectors:
      sum += np.array(vector)

    average = sum / len(vectors)
    return average