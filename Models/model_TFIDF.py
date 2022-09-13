import pandas as pd
import stanza
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import textwrap
import os

class model_TF_IDF:

  # konstruktor. pathToText är filvägen för patientberättelserna. amount är mängden patientberättelser
  def __init__(self, pathToText, amount):
    self.pathToText = pathToText
    self.amount = amount

  # skapar vektorrepresentationer
  def getVectors(self):

    # läser excelfilen med patientberättelserna. Patientberättelserna måste ligga under en kolumn som heter "Sammanfattning"
    data = pd.read_excel(self.pathToText, nrows=self.amount)
    stories = pd.DataFrame(data, columns=['Sammanfattning'])

    # förbehandlar texten i patientberättelserna och skapar vektorrepresentationer
    lemmaTokenizer = self.LemmaTokenizer()
    storyList = stories["Sammanfattning"].tolist()
    vectorizer = TfidfVectorizer(tokenizer = lemmaTokenizer)
    vectors = vectorizer.fit_transform(storyList)
    return vectors

  # klass för förbehandling av patientberättelserna
  class LemmaTokenizer(object):
    def __init__(self):
      self.nlp = stanza.Pipeline('sv', processors='tokenize, pos, lemma')
    def __call__(self, articles):
      doc = self.nlp(articles)    
      return list(word.lemma for sent in doc.sentences for word in sent.words if word.lemma not in string.punctuation)