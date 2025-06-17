import spacy

# Charger le modèle français de spaCy
nlp = spacy.load("fr_core_news_sm")

def semantic_chunking(text, max_chunk_size=150):
    doc = nlp(text)
    segments = []
    current_segment = []
    for sent in doc.sents:
        if len(" ".join(current_segment)) + len(sent.text) <= max_chunk_size:
            current_segment.append(sent.text)
        else:
            segments.append(" ".join(current_segment))
        current_segment = [sent.text]
        if current_segment:
            segments.append(" ".join(current_segment))
    return segments
    
# Exemple d'utilisation
text = "Le principal challenge du RAG est la gestion des documents très longs. En effet, la taille d'un document est généralement plus grande que le nombre de token maximal que peut prendre un LLM en entrée. Ceci implique l'utilisation des techniques qui permettent de découper le texte en des petits morceaux tout en préservant le contexte et la pertinence du document. Ce découpage est appelé Chunking. La stratégie de chunking influence votre système de RAG. Un mauvais découpage peut conduire à la perte de cohérence, perte d'information dans la mémoire non paramétrique de votre RAG."
segments = semantic_chunking(text, max_chunk_size=150)