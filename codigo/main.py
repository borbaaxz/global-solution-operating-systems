import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# CARREGAR DATASET

dados = pd.read_csv("../dataset/dataset_agricola.csv")

X = dados.drop("risco", axis=1)
y = dados["risco"]

# FLOAT64

X_float64 = X.astype(np.float64)

# Adicionar ruído para simular dados reais imperfeitos
np.random.seed(42)
ruido = np.random.normal(0, 5, X_float64.shape)
X_float64 = X_float64 + ruido

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_float64,
    y,
    test_size=0.2,
    random_state=42
)

modelo_float64 = Sequential()

modelo_float64.add(Dense(4, input_dim=5, activation='sigmoid'))

modelo_float64.add(Dense(1, activation='sigmoid'))

modelo_float64.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

inicio_float64 = time.time()

historico_float64 = modelo_float64.fit(
    X_treino,
    y_treino,
    epochs=20,
    verbose=0
)

fim_float64 = time.time()

tempo_float64 = fim_float64 - inicio_float64

loss_float64, acc_float64 = modelo_float64.evaluate(
    X_teste,
    y_teste,
    verbose=0
)

# INT8

X_int8 = X.astype(np.int8)

# Adicionar ruído também no int8
ruido_int8 = np.random.normal(0, 5, X_int8.shape)
X_int8 = X_int8 + ruido_int8
X_int8 = X_int8.astype(np.int8)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_int8,
    y,
    test_size=0.2,
    random_state=42
)

modelo_int8 = Sequential()

modelo_int8.add(Dense(4, input_dim=5, activation='sigmoid'))

modelo_int8.add(Dense(1, activation='sigmoid'))

modelo_int8.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

inicio_int8 = time.time()

historico_int8 = modelo_int8.fit(
    X_treino,
    y_treino,
    epochs=20,
    verbose=0
)

fim_int8 = time.time()

tempo_int8 = fim_int8 - inicio_int8

loss_int8, acc_int8 = modelo_int8.evaluate(
    X_teste,
    y_teste,
    verbose=0
)

# RESULTADOS

print("\n========== RESULTADOS ==========")

print(f"\nFLOAT64")
print(f"Accuracy: {acc_float64:.2f}")
print(f"Loss: {loss_float64:.2f}")
print(f"Tempo: {tempo_float64:.2f} segundos")

print(f"\nINT8")
print(f"Accuracy: {acc_int8:.2f}")
print(f"Loss: {loss_int8:.2f}")
print(f"Tempo: {tempo_int8:.2f} segundos")

# GRÁFICO COMPARATIVO - TEMPO

plt.figure(figsize=(8, 5))
tipos = ['float64', 'int8']
tempos = [tempo_float64, tempo_int8]
plt.bar(tipos, tempos)
plt.title('Tempo de Processamento')
plt.xlabel('Tipo Numérico')
plt.ylabel('Tempo (s)')
plt.grid(True)
plt.savefig('../graficos/benchmark.png')
plt.close()

# GRÁFICO ACCURACY

plt.figure(figsize=(8, 5))
plt.plot(historico_float64.history['accuracy'], label='Float64', marker='o')
plt.plot(historico_int8.history['accuracy'], label='Int8', marker='s')
plt.title('Accuracy por Época')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('../graficos/accuracy.png')
plt.close()

# GRÁFICO LOSS

plt.figure(figsize=(8, 5))
plt.plot(historico_float64.history['loss'], label='Float64', marker='o')
plt.plot(historico_int8.history['loss'], label='Int8', marker='s')
plt.title('Loss por Época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('../graficos/loss.png')
plt.close()

print("\nGráficos salvos em ../graficos/")
plt.show()