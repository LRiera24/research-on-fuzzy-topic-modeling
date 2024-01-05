from bs4 import BeautifulSoup
import os
from semantic_classsification import semantic_classification

def load_reuters_corpus(directory):
    """
    Carga el corpus de Reuters desde una serie de archivos SGML en el directorio especificado.
    Retorna una lista de strings, donde cada string es un documento.
    """
    documents = []

    # Recorre todos los archivos en el directorio
    for filename in os.listdir(directory):
        if filename.endswith('.sgm'):  # Asegúrate de que es un archivo SGML
            file_path = os.path.join(directory, filename)
            
            # Abre y lee el archivo
            with open(file_path, 'r', encoding='latin1') as file:
                content = file.read()

                # Usa BeautifulSoup para parsear el contenido SGML
                soup = BeautifulSoup(content, 'html.parser')

                # Encuentra todos los documentos
                for reuters in soup.find_all('reuters'):
                    # Extrae el texto del cuerpo del documento, si existe
                    body = reuters.find('body')
                    if body:
                        documents.append(body.text.strip())

    return documents

corpus = load_reuters_corpus(os.path.abspath('src') + '/corpus/reuters21578')
corpus_name = 'Reuters'
real_k = 135
real_tags = []

s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
c = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for sim in s:
    for coh in c:
        semantic_classification(corpus, corpus_name, real_k, real_tags, min_sim=sim, min_coh=coh, min_words_per_topic=100)