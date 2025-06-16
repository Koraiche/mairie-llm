import os
import numpy as np
from mistralai import Mistral
from mistralai.models import EmbeddingRequest
from sklearn.metrics.pairwise import cosine_similarity

#https://console.mistral.ai/api-keys
api_key = "EoOby7wNPyC2xCJDBATHb0By3JMK27b2"

# Initialisation du client avec la version 0.4.2
client = Mistral(api_key=api_key)
print("Client Mistral initialisé.")

# Textes pour lesquels nous générons des embeddings
textes_mistral = [
    "Quelles sont les heures d’ouverture de la mairie ?",
    "La mairie de Trifouillis-sur-Loire est ouverte du lundi au vendredi de 8h30 à 17h00.",
    "Le marché hebdomadaire a lieu tous les samedis matin sur la place centrale."
]

# Modèle d'embedding de Mistral
model_mistral = "mistral-embed"

print(f"\nGénération des embeddings avec le modèle '{model_mistral}'...")

# Génération des embeddings
response = client.embeddings.create(
    model=model_mistral,
    inputs=textes_mistral
)

# Extraction des embeddings de la réponse
embeddings_mistral_raw = response.data

# Conversion en array numpy pour faciliter les calculs
# (chaque objet a un attribut .embedding)
embeddings_mistral = np.array([item.embedding for item in embeddings_mistral_raw])

print("Embeddings Mistral générés !")
print("Shape du premier embedding Mistral :", embeddings_mistral[0].shape)

# --- Calcul de similarité ---
print("\n--- Calcul de Similarité (Mistral) ---")
embedding_question_mistral = embeddings_mistral[0]
embedding_doc1_mistral = embeddings_mistral[1]
embedding_doc2_mistral = embeddings_mistral[2]

similarity_q_doc1_mistral = cosine_similarity(
    [embedding_question_mistral],
    [embedding_doc1_mistral]
)[0][0]

similarity_q_doc2_mistral = cosine_similarity(
    [embedding_question_mistral],
    [embedding_doc2_mistral]
)[0][0]

print(f"Question : {textes_mistral[0]}")
print("*" * 30)
print(f"Document 1 : {textes_mistral[1]}")
print(f"Similarité avec la question : {similarity_q_doc1_mistral:.4f}")
print("*" * 30)
print(f"Document 2 : {textes_mistral[2]}")
print(f"Similarité avec la question : {similarity_q_doc2_mistral:.4f}")
print("*" * 30)

if similarity_q_doc1_mistral > similarity_q_doc2_mistral:
    print("\nConclusion (Mistral) : Le document 1 est sémantiquement plus proche de la question.")
else:
    print("\nConclusion (Mistral) : Le document 2 est sémantiquement plus proche de la question.")
