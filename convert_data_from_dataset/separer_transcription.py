import os

input_file = "transcripts.txt"

output_dir = "transcriptions"
os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            filename, text = line.strip().split("\t", 1)
            output_path = os.path.join(output_dir, f"{filename}.txt")
            with open(output_path, "w", encoding="utf-8") as out_file:
                out_file.write(text)
        except ValueError:
            print(f"Format invalide pour la ligne : {line.strip()}")