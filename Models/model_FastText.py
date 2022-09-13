import pandas as pd
import string
import stanza
import fasttext.util

class model_FastText:

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
    nlp = stanza.Pipeline('sv', processors='tokenize, pos, lemma')
    storyList = stories["Sammanfattning"].tolist()
    texts = []
    for story in storyList:
      texts.append(self.preProcess(story, nlp))

    # hämtar FastText-modellen
    fasttext.util.download_model('sv', if_exists='ignore')
    ft = fasttext.load_model('cc.sv.300.bin')

    # skapar vektorrepresentationer av patientberättelserna
    vectors = []
    for text in texts:
      vectors.append(ft.get_sentence_vector(text))
    
    return vectors

  # funktion för förbehandling
  def preProcess(self, text, nlp):
    doc = nlp(text)
    return " ".join(list(word.lemma for sent in doc.sentences for word in sent.words if word.lemma not in string.punctuation))



