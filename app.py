import redeNeural as redeNeural
import numpy as np
import matplotlib.pyplot as plt

entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas_esperadas = np.array([0, 1, 1, 0])

redeNeural = redeNeural.RedeNeuralMLP(tx_aprendizado=0.3)

print("Treinando rede neural")
redeNeural.treinar(entradas, saidas_esperadas, epocas=20000, tolerancia=0.005)

print("Treino concluído\n")

x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

pontos_grade = np.c_[xx.ravel(), yy.ravel()]
previsoes = []

for ponto in pontos_grade:
  _, resultado = redeNeural.forwardPass(ponto)
  previsoes.append(1 if resultado >= 0.5 else 0)

Z = np.array(previsoes).reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

for entrada, esperado in zip(entradas, saidas_esperadas):
  cor = 'red' if esperado == 1 else "blue"
  marcador = 'X' if esperado == 1 else 'o'
  
  plt.scatter(entrada[0], entrada[1], color=cor, marker=marcador, s=150, edgecolors='k', zorder=3, label=f"AND{entrada}={esperado}")

plt.title("Fronteira de Decisão do Neurônio (Porta Lógica XOR)", fontsize=14)
plt.xlabel("Entrada X1", fontsize=12)
plt.ylabel("Entrada X2", fontsize=12)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True, linestyle=":", alpha=0.5)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper left")

plt.show()