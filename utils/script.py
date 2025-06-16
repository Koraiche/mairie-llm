from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Déterminer l'appareil à utiliser (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

for i in range(100):
    # Preprocessing de l’input utilisateur
    question = input()
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Prédiction du modèle
    with torch.no_grad():
        output = model.generate(inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_new_tokens=50,
                                num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Question :", question)
    print("Réponse :", response)