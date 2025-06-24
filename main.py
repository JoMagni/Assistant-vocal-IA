import recognizer as recognizer # --> IA entrainé 
# import recognizer_google as recognizer # --> IA de google
import tts
from interpreter_ai.interpreter import interpret_command

def main():
    print("Assistant vocal lancé ✅")
    while True:
        print("En attente de la commande vocale...")
        command = recognizer.recognize_speech()

        if not command:
            continue

        print(f"Texte reconnu : {command}")

        response, action = interpret_command(command)

        print(f"Réponse de l'IA : {response}")

        tts.speak(response)

        if action == "exit":
            print("Assistant arrêté.")
            break


if __name__ == "__main__":
    main()