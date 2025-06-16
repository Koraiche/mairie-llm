from mistral.embedding import EmbeddingClient

# Initialisation du client
client = EmbeddingClient(api_key='YOUR_API_KEY')

# Texte à vectoriser
text = "Le chat dort sur le tapis."

# Obtention de l'embedding
embedding = client.get_embedding(text)

print(embedding)