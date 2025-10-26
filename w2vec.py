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

# En sentences se tiene que almacenar la información del dataset de noticias
# Cada sentencia puede ser una opinión
sentences = []

docs = []
count = 0
for item in sentences:
    docs.append(nlp(item))
    count += 1
corpus = [[x.text for x in y] for y in docs]
#print(corpus)

model = Word2Vec(corpus, min_count = 1, workers = 3, window = 10, sg = 1)

x_vals, y_vals, labels = reduce_dimensions(model)

plot_with_matplotlib(x_vals, y_vals, labels)