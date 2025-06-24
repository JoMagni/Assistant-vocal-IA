import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

with open("../data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

texts = []
labels = []

for intent in intents:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Modèle entraîné et sauvegardé !")
