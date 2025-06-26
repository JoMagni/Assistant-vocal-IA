import os
import subprocess
from pathlib import Path

def convert_opus_to_wav(source_dir, dest_dir):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for opus_path in source_dir.rglob('*.opus'):
        wav_filename = opus_path.stem + '.wav'
        wav_path = dest_dir / wav_filename
        
        command = [
            'ffmpeg',
            '-hide_banner', '-loglevel', 'error',
            '-i', str(opus_path),
            '-ar', '16000',
            '-ac', '1',
            str(wav_path)
        ]
        ²w
        try:
            subprocess.run(command, check=True)
            print(f"Converti: {opus_path} -> {wav_path}")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de la conversion de {opus_path}: {e}")

if __name__ == "__main__":
    source_directory = "/mon_fichier_d_entree"
    destination_directory = "/mon_fichier_d_arrivee"
    
    convert_opus_to_wav(source_directory, destination_directory)
    print(f"Conversion terminée! Tous les fichiers WAV sont dans: {destination_directory}")