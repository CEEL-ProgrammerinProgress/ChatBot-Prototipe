# Librerias de Math & Datos/Escritura/UI
import random
import json
import pickle
import streamlit as st
import numpy as np

# Librerias de Lenguaje Natural (NLP)
import nltk
from nltk.stem import WordNetLemmatizer

# Librerias de Deep Learning
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# lematizador (convierte palabras a su forma base, ej: "corriendo" -> "correr")
lemmatizer = WordNetLemmatizer()

with open("Parameters.json", "r", encoding="utf-8") as f:
    parameters = json.load(f)

#  paquetes necesarios de NLTK
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")

# Listas para almacenar nuestros datos estructurados.
words = []  # Todas las palabras únicas de los patrones.
classes = []  # Todos los tags únicos (saludo, despedida, etc.)
documents = []  # Combinación de palabras con su tag correspondiente.
ignore_letters = ["?", "!", "¿", ".", ","]  # Signos a ignorar.

# 1. PROCESAMIENTO DEL TEXTO.
# C
for intent in parameters["Parameters"]:
    for pattern in intent["patterns"]:
        # Tokenizar: divide la oración en palabras individuales
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Guardamos la oración tokenizada junto a su etiqueta (tag)
        documents.append((word_list, intent["tag"]))

        # Agregamos la etiqueta a nuestras clases si no existe aún
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lematizamos las palabras, las pasamos a minúsculas y quitamos duplicados (set)
words = [
    lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters
]
words = sorted(set(words))
classes = sorted(set(classes))

# Guardamos las palabras y clases en archivos .pkl para usarlos luego (se usanran en las respuestas)
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# 2. PREPARACIÓN DE LOS DATOS DE ENTRENAMIENTO
# E
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Creamos la "Bolsa de palabras" (Bag of Words): 1 si la palabra está, 0 si no
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Creamos un array de salida (1 para el tag actual, 0 para los demás)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Mezclamos los datos de entrenamiento para que la red neuronal aprenda mejor
random.shuffle(training)

# CORRECCIÓN: Separación correcta de características (X) y etiquetas (Y) para evitar errores de NumPy
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# 3. CREACIÓN DE LA RED NEURONAL (MODELO Keras)
# E
model = Sequential()
# Capa de entrada con 128 neuronas. ReLu es la función de activación.
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
# Dropout apaga el 50% de las neuronas aleatoriamente para evitar sobreajuste (overfitting)
model.add(Dropout(0.5))
# Capa oculta con 64 neuronas
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
# Capa de salida con tantas neuronas como tags tengamos. Softmax nos da probabilidades (ej: 90% saludo)
model.add(Dense(len(train_y[0]), activation="softmax"))

# CORRECCIÓN: Configuración del optimizador y corrección del typo 'opmizer'
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Entrenamiento del modelo
train_process = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Guardado correcto del modelo (solo requiere el nombre del archivo)
model.save(
    "Bit_Chatbot_RRH.keras"
)  # <- AQUI HAY QUE PONER COMO QUIERSES QUE SE LLAME EL BOT
print("¡Modelo entrenado y guardado con éxito!")
# L
