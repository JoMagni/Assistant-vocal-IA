import speech_recognition as sr

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Parle...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="fr-FR")
        print(f"Tu as dit : {text}")
        return text
    except sr.UnknownValueError:
        print("Je n'ai pas compris.")
        return ""
    except sr.RequestError as e:
        print(f"Erreur de service : {e}")
        return ""