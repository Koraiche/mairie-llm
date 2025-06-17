from bs4 import BeautifulSoup

def chunk_by_tags(html_content, tags):
    soup = BeautifulSoup(html_content, 'html.parser')
    segments = []
    for tag in tags:
        elements = soup.find_all(tag)
        for elem in elements:
            segments.append(elem.get_text())
    return segments

# Exemple d'utilisation
html_content = "<h1>Titre</h1><p>Paragraphe 1</p><p>Paragraphe 2</p>"
tags = ['h1', 'p']
segments = chunk_by_tags(html_content, tags)

print(segments)