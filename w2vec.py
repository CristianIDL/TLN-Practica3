import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import spacy
import string
from gensim.models import Word2Vec
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

charfilter = re.compile('[a-zA-Z]+')
nltk.download('stopwords')
stopwords_es = set(stopwords.words('spanish'))
# !python -m spacy download es_core_news_sm
nlp = spacy.load("es_core_news_sm")

def tokenize(document, word):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document)
    tokens = [token for token in tokens if token not in stopwords_es and token.isalpha()]
    tokens = [token for token in tokens if token != word]
    return set(tokens)

def preprocessing(sent):
    # Eliminar signos de puntuación
    #words = sent.translate(str.maketrans('', '', string.punctuation))
    words = re.sub(r'[\.\,"\'-?:!;]+', '', sent)
    words = sent.split()
    word_lower = []
    for word in words:
        word_lower.append(word.lower())
    # Quitar stop words:
    word_clean = [word for word in word_lower if word not in stopwords_es]
    # Sólo caracteres
    tokens = list(filter(lambda token : charfilter.match(token),word_clean))
    # Stemming
    ntokens = []
    for word in tokens:
        ntokens.append(PorterStemmer().stem(word))
    return ' '.join(tokens)

# Código para reducir dimensionalidad y que se puedan graficar las palabras
def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

# Función para graficar
def plot_with_matplotlib(x_vals, y_vals, labels):
    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.title('Tsne Visualización')
    plt.show()

# CARGAR EL DATASET CSV
try:
    # Cargar el dataset
    df = pd.read_csv('dataset_limpio.csv')
    print(f"Dataset cargado con {len(df)} filas")
    print("Columnas disponibles:", df.columns.tolist())

    # Creación del corpus a partir de la columna 'detalles_limpios' 
    sentences = df['detalles_limpios'].dropna().tolist()
    
    print(f"Número de oraciones cargadas: {len(sentences)}")
    print("\nPrimeras 5 oraciones:")
    for i, sent in enumerate(sentences[:5]):
        print(f"{i+1}: {sent}")
        
except FileNotFoundError:
    print("Error: No se encontró el archivo 'dataset_limpio.csv'")
    print("Asegúrate de que el archivo esté en el mismo directorio")
    sentences = []
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    sentences = []

# PROCESAR EL CORPUS
docs = []
count = 0

print("\nProcesando documentos con spaCy...")
for item in sentences:
    # Si la columna ya está tokenizada (como lista de strings), unir los tokens
    if isinstance(item, list):
        item = ' '.join(item)
    elif isinstance(item, str) and '[' in item and ']' in item:
        # Si parece una lista en formato string, intentar limpiarla
        item = re.sub(r'[\[\]\'\"]', '', item)
    
    doc = nlp(str(item))  # Asegurar que sea string
    docs.append(doc)
    count += 1
    
    # Mostrar progreso cada 100 documentos
    if count % 100 == 0:
        print(f"Procesados {count} documentos...")

print(f"Total de documentos procesados: {count}")

# Crear el corpus para Word2Vec
corpus = []
for doc in docs:
    # Extraer tokens que no sean stopwords, sean alfabéticos y tengan más de 1 carácter
    doc_tokens = [token.text.lower() for token in doc 
                 if not token.is_stop and token.is_alpha and len(token.text) > 1]
    if doc_tokens:  # Solo agregar si hay tokens
        corpus.append(doc_tokens)

print(f"Tamaño del corpus: {len(corpus)} documentos")
print(f"Total de tokens únicos: {len(set([token for doc in corpus for token in doc]))}")

# Entrenar el modelo Word2Vec
if len(corpus) > 0:
    print("\nEntrenando modelo Word2Vec...")
    # Parámetros para la función Word2Vec:
        # min_count = 2: Ignora palabras que solo salen una vez
        # workers = 3: Usar 3 hilos para entrenar el modelo
        # window = 5: Contexto de 5 palabras a la izquierda y derecha
        # sg = 1: Usar skip-gram (en lugar de CBOW)
        # vector_size = 100: Tamaño de los vectores de palabras
    model = Word2Vec(corpus, min_count = 2, workers = 3, window = 5, sg = 1, vector_size = 100)

    print(f"Vocabulario del modelo: {len(model.wv.key_to_index)} palabras")
    print("\nEjemplo de palabras similares:")
    
    # Mostrar algunas palabras similares como ejemplo
    if len(model.wv.key_to_index) > 0:
        try:
            example_word = list(model.wv.key_to_index.keys())[0]
            similar_words = model.wv.most_similar(example_word, topn=5)
            print(f"Palabras similares a '{example_word}':")
            for word, score in similar_words:
                print(f"  {word}: {score:.3f}")
        except:
            print("No se pudieron calcular similitudes para la primera palabra")
    
    # Reducir dimensiones y graficar
    print("\nReduciendo dimensiones con t-SNE...")
    x_vals, y_vals, labels = reduce_dimensions(model)
    
    print("Generando gráfico...")
    plot_with_matplotlib(x_vals, y_vals, labels)
    
else:
    print("No hay suficiente datos en el corpus para entrenar el modelo")