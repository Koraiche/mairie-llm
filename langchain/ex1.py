from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialisation du splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, # Taille de chaque segment
    chunk_overlap=20) # Chevauchement entre les segments
    
# Découpage du texte
segments = text_splitter.split_text("Le principal challenge du RAG est la gestion des documents très longs. En effet, la taille d'un document est généralement plus grande que le nombre de token maximal que peut prendre un LLM en entrée. Ceci implique l'utilisation des techniques qui permettent de découper le texte en des petits morceaux tout en préservant le contexte et la pertinence du document. Ce découpage est appelé Chunking. La stratégie de chunking influence votre système de RAG. Un mauvais découpage peut conduire à la perte de cohérence, perte d'information dans la mémoire non paramétrique de votre RAG.")

print(segments)