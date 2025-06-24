# Assistant vocal IA

## project download
git clone https://github.com/JoMagni/Assistant-vocal-IA.git

## Start project
Open a console
Do a 'cd' commande to go into the project

--> You can go in step 4 if you didn't whant train AIs

### 1 - Install requirements
pip install -r requirements.txt

### 2 - Train STT model
cd .\audio_ai\
python .\train_audio_model.py

### 3 - Train NLP
cd ..\interpreter_ai\
python .\train_model.py
cd ..

### 4 - Start Programme
python .\main.py