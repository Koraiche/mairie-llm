# pymupdf4llm
#!pip install pymupdf4llm --quiet
import pymupdf4llm

# Ouvrir le fichier PDF
md_text = pymupdf4llm.to_markdown("./sample.pdf")
print("=== Contenu Markdown extrait avec PyMuPDF4LLM ===")
print(md_text)