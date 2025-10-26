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
nlp = spacy.load("es_core_news_sm")

def tokenize(document, word):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document)
    tokens = [token for token in tokens if token not in stopwords_es and token.isalpha()]
    tokens = [token for token in tokens if token != word]
    return set(tokens)

def preprocessing(sent):
    # Eliminar signos de puntuación
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

    sentences = df['detalles_limpios'].dropna().tolist()
    print(f"Número de oraciones cargadas: {len(sentences)}")
    
    print("\nPrimeras 5 oraciones:")
    for i, sent in enumerate(sentences[:5]):
        print(f"{i+1}: {sent}")
        
except FileNotFoundError:
    print("Error: No se encontró el archivo con el dataset 'dataset_limpio.csv'.")
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
    model = Word2Vec(corpus, 
                     min_count = 2, # min_count = 2: Ignora palabras que solo salen una vez
                     workers = 3, # workers = 3: Usar 3 hilos para entrenar el modelo
                     window = 5, # window = 5: Contexto de 5 palabras a la izquierda y derecha
                     sg = 1, # sg = 1: Usar skip-gram (en lugar de CBOW)
                     vector_size = 100) # vector_size = 100: Tamaño de los vectores de palabras

    print(f"Vocabulario del modelo: {len(model.wv.key_to_index)} palabras")

    # --- ANÁLISIS DE DISTRIBUCIÓN DE DATOS ---

    def analizar_distribucion_datos(corpus, model):
        """
        Análisis completo de la distribución del corpus y el modelo Word2Vec
        """
        print("=" * 60)
        print("ANÁLISIS DE DISTRIBUCIÓN DE DATOS")
        print("=" * 60)
        
        # 1. ESTADÍSTICAS BÁSICAS DEL CORPUS
        print("\n1. ESTADÍSTICAS BÁSICAS DEL CORPUS")
        print("-" * 40)
        
        total_documentos = len(corpus)
        total_palabras = sum(len(doc) for doc in corpus)
        palabras_unicas = len(set([palabra for doc in corpus for palabra in doc]))
        vocabulario_modelo = len(model.wv.key_to_index)
        
        print(f"Total de documentos: {total_documentos}")
        print(f"Total de palabras: {total_palabras}")
        print(f"Palabras únicas en corpus: {palabras_unicas}")
        print(f"Vocabulario del modelo: {vocabulario_modelo}")
        print(f"Promedio de palabras por documento: {total_palabras/total_documentos:.2f}")
        
        # 2. DISTRIBUCIÓN DE LONGITUD DE DOCUMENTOS
        print("\n2. DISTRIBUCIÓN DE LONGITUD DE DOCUMENTOS")
        print("-" * 40)
        
        longitudes = [len(doc) for doc in corpus]
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(longitudes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(longitudes), color='red', linestyle='--', label=f'Media: {np.mean(longitudes):.2f}')
        plt.axvline(np.median(longitudes), color='orange', linestyle='--', label=f'Mediana: {np.median(longitudes):.2f}')
        plt.xlabel('Palabras por documento')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Longitud de Documentos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. ANÁLISIS DE FRECUENCIA DE PALABRAS
        print("\n3. ANÁLISIS DE FRECUENCIA DE PALABRAS")
        print("-" * 40)
        
        # Calcular frecuencia de todas las palabras
        from collections import Counter
        todas_palabras = [palabra for doc in corpus for palabra in doc]
        frecuencias = Counter(todas_palabras)
        
        # Palabras más frecuentes
        palabras_comunes = frecuencias.most_common(20)
        
        print("- 20 palabras más frecuentes:")
        for i, (palabra, freq) in enumerate(palabras_comunes, 1):
            print(f"  {i:2d}. {palabra:<15} : {freq:>4} ocurrencias")
        
        # Gráfico de palabras más frecuentes
        plt.subplot(1, 3, 2)
        palabras, counts = zip(*palabras_comunes)
        plt.barh(range(len(palabras)), counts, color='lightgreen')
        plt.yticks(range(len(palabras)), palabras)
        plt.xlabel('Frecuencia')
        plt.title('20 Palabras Más Frecuentes')
        plt.gca().invert_yaxis()
        
        # 4. DISTRIBUCIÓN DE FRECUENCIA (Ley de Zipf)
        print("\n4. DISTRIBUCIÓN DE FRECUENCIA (Ley de Zipf)")
        print("-" * 40)
        
        frecuencias_ordenadas = sorted(frecuencias.values(), reverse=True)
        
        plt.subplot(1, 3, 3)
        plt.loglog(range(1, len(frecuencias_ordenadas) + 1), frecuencias_ordenadas, 'b-', alpha=0.7)
        plt.xlabel('Rank (log)')
        plt.ylabel('Frecuencia (log)')
        plt.title('Ley de Zipf - Distribución de Frecuencias')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 5. ANÁLISIS DE RANGO DE FRECUENCIAS
        print("\n5. ANÁLISIS DE RANGO DE FRECUENCIAS")
        print("-" * 40)
        
        freq_ranges = {
            '1 ocurrencia': sum(1 for f in frecuencias.values() if f == 1),
            '2-5 ocurrencias': sum(1 for f in frecuencias.values() if 2 <= f <= 5),
            '6-20 ocurrencias': sum(1 for f in frecuencias.values() if 6 <= f <= 20),
            '21-100 ocurrencias': sum(1 for f in frecuencias.values() if 21 <= f <= 100),
            'Más de 100 ocurrencias': sum(1 for f in frecuencias.values() if f > 100)
        }
        
        for rango, cantidad in freq_ranges.items():
            porcentaje = (cantidad / palabras_unicas) * 100
            print(f"• {rango:<20}: {cantidad:>4} palabras ({porcentaje:.1f}%)")
        
        # 6. PALABRAS CON MEJOR REPRESENTACIÓN EN EL MODELO
        print("\n6. PALABRAS CON MAYOR FRECUENCIA EN EL MODELO")
        print("-" * 40)
        
        # Obtener palabras ordenadas por frecuencia en el modelo
        palabras_modelo = model.wv.key_to_index
        palabras_ordenadas = sorted(palabras_modelo.keys(), 
                                key=lambda x: frecuencias.get(x, 0), 
                                reverse=True)[:15]
        
        for i, palabra in enumerate(palabras_ordenadas, 1):
            freq = frecuencias.get(palabra, 0)
            print(f"  {i:2d}. {palabra:<15} : {freq:>4} ocurrencias")
        
        return frecuencias, longitudes

    # --- ANÁLISIS DE CALIDAD DEL MODELO WORD2VEC ---

    def analizar_calidad_modelo(model, corpus, frecuencias):
        """
        Análisis de la calidad del modelo Word2Vec
        """
        print("\n" + "=" * 60)
        print("ANÁLISIS DE CALIDAD DEL MODELO WORD2VEC")
        print("=" * 60)
        
        # 1. ESTADÍSTICAS DEL MODELO
        print("\n1. ESTADÍSTICAS DEL MODELO")
        print("-" * 40)
        
        vocab_size = len(model.wv.key_to_index)
        vector_size = model.wv.vector_size
        total_vectores = vocab_size * vector_size
        
        print(f"Tamaño del vocabulario: {vocab_size}")
        print(f"Dimensionalidad de vectores: {vector_size}")
        print(f"Total de parámetros: {total_vectores:,}")
        print(f"Arquitectura: {'Skip-gram' if model.sg == 1 else 'CBOW'}")
        print(f"Ventana de contexto: {model.window}")
        
        # 2. COBERTURA DEL VOCABULARIO
        print("\n2. COBERTURA DEL VOCABULARIO")
        print("-" * 40)
        
        palabras_corpus = set([palabra for doc in corpus for palabra in doc])
        palabras_modelo = set(model.wv.key_to_index.keys())
        
        cobertura = len(palabras_modelo) / len(palabras_corpus) * 100
        print(f"Palabras en corpus: {len(palabras_corpus)}")
        print(f"Palabras en modelo: {len(palabras_modelo)}")
        print(f"Cobertura del modelo: {cobertura:.1f}%")
        
        # 3. EJEMPLOS DE SIMILITUD SEMÁNTICA
        print("\n3. PRUEBAS DE SIMILITUD SEMÁNTICA")
        print("-" * 40)
        
        # Seleccionar algunas palabras frecuentes para probar
        palabras_frecuentes = [palabra for palabra, _ in frecuencias.most_common(50)]
        palabras_a_probar = palabras_frecuentes[:10]  # Probar con las 10 más frecuentes
        
        for palabra in palabras_a_probar:
            if palabra in model.wv:
                try:
                    similares = model.wv.most_similar(palabra, topn=5)
                    print(f"\nPalabras similares a '{palabra}':")
                    for similar, score in similares:
                        print(f"    {similar}: {score:.3f}")
                except:
                    print(f"\nNo se pudieron encontrar similares para '{palabra}'")
        
        # 4. ANÁLISIS DE VECINOS MÁS CERCANOS POR FRECUENCIA
        print("\n4. ANÁLISIS POR RANGO DE FRECUENCIA")
        print("-" * 40)
        
        # Palabras de alta frecuencia
        palabras_alta_freq = [p for p, f in frecuencias.most_common(20) if f > 10]
        # Palabras de media frecuencia
        palabras_media_freq = [p for p, f in frecuencias.most_common(100) if 5 <= f <= 10][:5]
        # Palabras de baja frecuencia
        palabras_baja_freq = [p for p, f in frecuencias.items() if f == 1][:5]
        
        categorias = [
            ("Alta frecuencia", palabras_alta_freq[:3]),
            ("Media frecuencia", palabras_media_freq[:3]),
            ("Baja frecuencia", palabras_baja_freq[:3])
        ]
        
        for categoria, palabras in categorias:
            print(f"\n• {categoria}:")
            for palabra in palabras:
                if palabra in model.wv:
                    try:
                        similares = model.wv.most_similar(palabra, topn=3)
                        similares_str = ", ".join([f"{s}({score:.2f})" for s, score in similares])
                        print(f"    {palabra} -> {similares_str}")
                    except:
                        print(f"    {palabra} -> No se pudieron calcular similitudes")

    # =============================================================================
    # EJECUTAR ANÁLISIS
    # =============================================================================

    # Ejecutar análisis si el modelo fue entrenado correctamente
    if 'model' in locals() and len(corpus) > 0:
        print("INICIANDO ANÁLISIS COMPLETO...")
        
        # Análisis de distribución de datos
        frecuencias, longitudes = analizar_distribucion_datos(corpus, model)
        
        # Análisis de calidad del modelo
        analizar_calidad_modelo(model, corpus, frecuencias)
        
        print("\n" + "=" * 60)
        print("ANÁLISIS COMPLETADO")
        print("=" * 60)
        
    else:
        print("No se pudo ejecutar el análisis - Modelo no entrenado o corpus vacío")

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