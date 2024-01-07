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
    topics = []
    file_path = os.path.join(directory, "all-topics-strings.lc.txt")
    with open(file_path, 'r') as file:
        for line in file:
            # Strip newline and whitespace characters and add to the list
            topics.append(line.strip())
    return documents, topics

corpus, real_tags = load_reuters_corpus(os.path.abspath('src') + '/corpus/reuters21578')
corpus_name = 'Reuters'
real_k = 135

description = 'fase1'
min_words_per_topic = 50

test_folder = os.path.abspath('tests')
test_folder += f'/{corpus_name}'

if not os.path.exists(test_folder):
    os.makedirs(test_folder)

print(test_folder)
# Get the count of files in the folder
file_count = len([name for name in os.listdir(test_folder)])
print(file_count)
test_folder += f'/run{file_count+1}_{min_words_per_topic}'

if description:
    description = description.lower()
    description = description.split(' ')
    description = '_'.join(description)
    test_folder += f'_{description}'

os.makedirs(test_folder)

s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
c = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for sim in s:
    for coh in c:
        semantic_classification(corpus, corpus_name, real_k, real_tags, test_folder=test_folder, min_sim=sim, min_coh=coh, min_words_per_topic=min_words_per_topic)
