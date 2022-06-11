import tensorflow as tf  # Librería de inteligencia artifical
import numpy as np       # Librería para arreglos numéricos

celsius = np.array([-40, 10, 100, 58, 32], dtype=float)         # Entradas
fahrenheit = np.array([-40, 50, 212, 136.4, 89.6], dtype=float) # Salidas


# Framework Keras
capa = tf.keras.layers.Dense(units=1, input_shape=[1])  # Capa densa
modelo = tf.keras.Sequential([capa])

"""
# Se agregan capas ocultas (esto significa más capacidad de expresión)
# Para probar esto comentar las dos lineas anteriores)
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])
"""
# Peso: importancia de la conexión entre neuronas
# Sesgo: controla que tan volátil es el resultado de la neurona independientemente del peso
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),   # Algoritmo para ajustar pesos y sesgos
    loss = 'mean_squared_error'
)

print("Entrenando...\n")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Listo !\n")

import matplotlib.pyplot as plt

plt.xlabel(" Evolución de aprendizaje ")
plt.ylabel(" Cantidad de errores ")
plt.plot(historial.history["loss"])

celsiusPrueba = 70.0
print("\n", str(celsiusPrueba), "Celsius son aproximadamente ", str(modelo.predict([celsiusPrueba])), " Fahrenheit\n")
"""
print("Variables internas del modelo: \n")
print(capa.get_weights(),"\n")
"""
print("\n Se mide la cantidad de errores a medida que va aprendiendo\n")
