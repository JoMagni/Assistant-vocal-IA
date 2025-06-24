import joblib
import os
from datetime import datetime

CURRENT_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(CURRENT_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(CURRENT_DIR, "vectorizer.pkl"))

def interpret_command(command):
    prediction = model.predict(vectorizer.transform([command]))[0]
    print(f"Prédiction : {prediction}")

    if prediction == "heure":
        return f"Il est {datetime.now().strftime('%H:%M')}", None

    elif prediction == "jour":
        return f"Aujourd'hui, nous sommes le {datetime.now().strftime('%A %d %B %Y')}", None

    elif prediction == "blague":
        return "Pourquoi les programmeurs n'aiment-ils pas la nature ? Parce qu'il y a trop de bugs.", None

    elif prediction == "exit":
        return "D'accord, j'arrête.", "exit"

    else:
        return "Je n'ai pas compris la commande.", None