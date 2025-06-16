from sentence_transformers import SentenceTransformer
import numpy as np

# Chargement du modèle SBERT
model_hf = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

texts_hf = [
    "Quelles sont les heures d'ouvertures de la mairie ?",
    "La mairie de Trifouillis-sur-Loire est ouverte du lundi au vendredi de 8h30 à 17h30.",
    "Le marché hebdomadaire a lieu tous les samedis matin sur la place centrale."
]

embeddings_hf = model_hf.encode(texts_hf)

print(embeddings_hf.shape)

def cosine_similarity(vec1, vec2): 
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

embedding_question = embeddings_hf[0]

embeddings_doc1 = embeddings_hf[1]
embeddings_doc2 = embeddings_hf[2]

similarity_doc1 = cosine_similarity(embedding_question, embeddings_doc1)
similarity_doc2 = cosine_similarity(embedding_question, embeddings_doc2)


print(f"Question : {texts_hf[0]}")
print("--------------------------------")
print(f"Document 1 : {texts_hf[1]}")
print(f"Similarité avec le document 1 : {similarity_doc1}")


print("--------------------------------")
print(f"Document 2 : {texts_hf[2]}")
print(f"Similarité avec le document 2 : {similarity_doc2}")


if similarity_doc1 > similarity_doc2:
    print("Le document 1 est plus similaire à la question.")
else:
    print("Le document 2 est plus similaire à la question.")







