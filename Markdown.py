# markitdown
#!pip install markitdown
from markitdown import MarkItDown

# Ouvrir le fichier PDF
md = MarkItDown()
result = md.convert("./sample.pdf")
print(result.text_content)