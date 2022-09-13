import pandas as pd
import sys
import textwrap
import umap
import plotly.express as px
from model_BERT import model_BERT
from model_LASER import model_LASER
from model_FastText import model_FastText
from model_TFIDF import model_TF_IDF

# Metod för att spara vektorer i en excelfil. Tar in ett filnamn: exempel.xlsx.
def saveFile(path, vectors):
  writer = pd.ExcelWriter(f"resources/{path}")
  df = pd.DataFrame(vectors)
  df.to_excel(writer)
  writer.save()

# Metod för att läsa excel-fil med sparade vektorrepresentationer. 
# path är filvägen: resources/exempel.xlsx och rows är antal rader som ska läsas.
def readExistingFile(path, rows):
  df = pd.read_excel(path, nrows=rows)
  df.drop(df.columns[0], axis=1, inplace=True)
  return pd.DataFrame(df)

# Metod för att reducera vektorerna till 2 dimensioner. 
# De tre första parametrarna motsvara umap-parametrarna: neighbors, min_dist och n_components. vectors är vektorerna som ska reduceras.
# Läs mer om parametrarna här: https://umap-learn.readthedocs.io/en/latest/parameters.html
def reduce(neighbors, dist, components, vectors):
  UMAP_results = umap.UMAP(n_neighbors = neighbors, min_dist = dist, n_components = components, metric = 'euclidean').fit_transform(vectors)
  UMAP_df = pd.DataFrame(data=UMAP_results, columns=['X', 'Y'])
  return UMAP_df

# Fabriksmetod
def Model(model, path, amount):

  models = {
    "BERT": model_BERT,
    "LASER": model_LASER,
    "TF-IDF": model_TF_IDF,
    "FastText": model_FastText,
  }

  return models[model](path, amount)

def main():
  # UMAP parametrar. Läs mer om dessa här https://umap-learn.readthedocs.io/en/latest/parameters.html:
  neighbors = 10
  dist = 0.01
  components = 2

  # Boolean för att dela upp datapunkterna i olika färger baserat på patientberättelsernas huvudproblem
  showColor = True

  # Filvägen till excel-filen med patientberättelser.
  # Behöver innehålla kolumner med namnen: Sammanfattning (kolumnen innehåller patientberättelserna i textform)
  # och Huvudproblem (Innehåller huvudproblemen)
  pathToStories = ""

  # instansiering av variabler
  model = ""
  save = False
  fileName = ""
  vectorDF = []
  visualize = ""

  # Prompt för om färdiga vektorer(redan sparade vektorer i excel) ska användas, 
  # tar in strängen "true" eller "false", allt annat avslutar programmet.
  existing = input("Skriv true för att använda färdiga vektorer, annars skriv false\n")
  # Promt som tar in antalet rader(patientberättelser) som ska behandlas.
  amount = int(input("Ange antalet berättelser/vektorer att hantera\n"))

  if existing == "true":
    # Prompt som tar in filnamnet på excel-filen med redan sparade vektorer. Filen måste ligga i resources med formatet: exempel.xlsx.
    fileName = input("Ange namnet på .xlsx-filen (med ändelse)\n")
    print("Skapar spridningsdiagram...")
    vectorDF = readExistingFile(f"resources/{fileName}", amount)

  elif existing == "false":
    # Val av vektorrepresentationsmodell
    model = str(input("Ange vilken modell som ska användas: BERT, LASER, TF-IDF eller FastText\n"))
    # Stänger programmet om input ej stämmer överrens med valen.
    if model not in {"BERT", "LASER", "TF-IDF", "FastText"}:
      print("Den angivna modellen finns inte. Stänger programmet...")
      sys.exit(0)

    # Prompt som tar in huruvida vektorrepresentationerna ska sparas som en excelfil eller ej. 
    # Vektorerna sparas endas om input är strängen "true"
    save = input("Skriv true för att spara vektorerna, annars skriv false\n")
 
    if save == "true":
      # Prompts som tar in filnamnet excel-filen får. filnamnet ska vara på formen: exempel.xlsx.
      fileName = input("Ange namnet på .xlsx-filen (med .xlsx)\n")
      # Prompt som tar in huruvida vektorerrepresentationerna ska visualiseras efter att de sparats eller ej. 
      # Vektorerna visualiseras om input är strängen "true", all annan input avslutar programmet.
      visualize = input("Skriv true för att skapa spridningsdiagram, annars skriv false\n")

    #Skapar vektorrepresentationer
    vectorModel = Model(model, pathToStories, amount)
    print("Skapar vektorer...")
    vectorDF = vectorModel.getVectors()
  
  # Input till variabeln existing var varken "true" eller "false"
  else:
    print("Input om färdiga vektorer ska användas eller ej var ej korrekt.\n Avslutar...")
    sys.exit(0)
  
  if save == "true":
    saveFile(fileName, vectorDF)

    if visualize != "true":
        sys.exit(0)
    
    print("skapar spridningsdiagram...")

  # Läser in filen med patientberättelser.
  data = pd.read_excel(pathToStories, nrows=amount)

  # Skapar lista med berättelsernas huvudproblem samt en lista med patientberättelserna i textform som behandlas med linebreak.
  labels = list(data.Huvudproblem)
  hoverShow = list()
  for x in range(amount):
   hoverShow.append(textwrap.fill(data["Sammanfattning"][x], width = 60).replace('\n', '<br />'))

  # Reducerar vektorerna till 2 dimensioner och lägger till kolumner för patientberättelserna i textform och huvudproblemen. 
  reducedVectors = reduce(neighbors, dist, components, vectorDF)
  reducedVectors['T'] = hoverShow
  reducedVectors['L'] = labels

  # Skapar spridningsdiagram med färg för att kategorisera datapunkterna efter patientberättelsens huvudproblem.
  if showColor:
    fig = px.scatter(reducedVectors, x = 'X', y = 'Y', color = 'L', hover_data = 'T', title = f'{model}: {amount} stycken')
    fig.show()

  # Skapar spridningsdiagram med bara en färg på datapunkterna.
  else:
    fig = px.scatter(reducedVectors, x = 'X', y = 'Y', hover_data = 'T', title = f'{model}: {amount} stycken')
    fig.show()

if __name__ == "__main__":
  main() 