# Användarmanual

Denna användarmanual är ett pågående arbete och kan komma att uppdateras. Exempelvis ska det länkas till rapporten när den reviderats färdigt.

Utöver denna manual finns det hjälpsamma kommentarer i kodfilerna.

Allting i detta repo har bara testats på Windows-datorer. Inga garantier att allt skulle funka likadant med andra operativsystem.

# Instruktioner

- Klona detta Git-repo.
- Ladda ner Python. Använd den senaste versionen om möjligt. (https://www.python.org/downloads/)
- Öppna kommandotolken och navigera dig dit Git-repot har klonats.
- De flesta dependencies kan hämtas genom att skriva i kommandotolken “py -m pip install requirements.txt”
- FastText är inte del av requirements.txt. Hämta FastText enligt instruktionerna i länken: https://fasttext.cc/docs/en/support.html#building-fasttext-python-module. Om requirements som nämns i länken inte uppfylls går det att hämta en wheel-fil direkt från följande länk: https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext. Kör sedan filen med hjälp av kommandot “py -m pip install *filnamnet.whl*”
- Stanza är inte del av requirements.txt. Ladda ned det genom att skriva “py -m pip install stanza” i kommandotolken.

# Beskrivning

I detta repo finns det två stycken mappar med kod. 

## Models
Här hittar ni koden för att skapa spridningsdiagram. Spridningsdiagram skapas genom att exekvera mainmetoden. Vid exekvering kommer man få ett antal prompts i konsolen så att man kan bestämma hur spridningsdiagrammen ska se ut. Se mer om detta i kommentarerna i koden.

## Model Evaluation
Här hittar ni fyra exekverbara filer som ger ett värde på Pearson- och Spearmans-korrelationskoefficient för varje modell. Se rapporten för vidare information om detta.
