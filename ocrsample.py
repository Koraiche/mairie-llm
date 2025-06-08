# Easy OCR
#!pip install easyocr --quiet
import easyocr
import os

# Initialiser EasyOCR
reader = easyocr.Reader(['fr'])
# 'fr' pour le français, ajoutez d'autres langues si nécessaire

# Extraire le texte de chaque image
full_text = ""

# Appliquer l'OCR sur l'image
results = reader.readtext("./image.png")

# Formater les résultats
for (bbox, text, prob) in results:
    full_text += text + "\n"
    
print(full_text)