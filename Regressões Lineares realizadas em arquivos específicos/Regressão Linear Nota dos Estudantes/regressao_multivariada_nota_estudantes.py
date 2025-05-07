import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

def column(matrix, i):
    return [float(row[i]) for row in matrix]

def h(x, x2, te0, te1, te2):
    return te0 + te1*x + te2*x2

dados = pd.read_csv('Student_Marks.csv', sep=',')
dados.head()

c1 = column(dados.values, 0)
c2 = column(dados.values, 1)
c3 = column(dados.values, 2)

def regressao(c1, c2, c3, alpha=0.01, epochs=10000):
    t0, t1, t2 = 0.0, 1.0, 1.0
    m = len(c1)
    for passo in range(epochs):
        somaZero = 0.0
        somaUm = 0.0
        somaDois = 0.0

        for i in range(m):
            erro = h(c1[i], c2[i], t0, t1, t2) - c3[i]
            somaZero += erro
            somaUm += erro * c1[i]
            somaDois += erro * c2[i]

        t0 -= alpha * (somaZero / m)
        t1 -= alpha * (somaUm / m)
        t2 -= alpha * (somaDois / m)

    return t0, t1, t2

kf = KFold(n_splits=5, shuffle=True, random_state=42)
erros = []

for conjunto_treino, conjunto_teste in kf.split(c1):

    c1_treino, c1_teste = np.array(c1)[conjunto_treino], np.array(c1)[conjunto_teste]
    c2_treino, c2_teste = np.array(c2)[conjunto_treino], np.array(c2)[conjunto_teste]
    esperado_treino, esperado_teste = np.array(c3)[conjunto_treino], np.array(c3)[conjunto_teste]

    t0, t1, t2 = regressao(c1_treino, c2_treino, esperado_treino)

    previsao = [h(x1, x2, t0, t1, t2) for x1, x2 in zip(c1_teste, c2_teste)]

    erro = np.mean((np.array(esperado_teste) - np.array(previsao))**2)
    erros.append(erro)

m = len(c1)
yh = [h(c1[i], c2[i], t0, t1, t2) for i in range(m)]

print('R²:', 1 - (np.sum((np.array(c3) - np.array(yh))**2) / np.sum((np.array(c3) - np.mean(c3))**2)))
print("Valor Previsão:", h(1, 2, t0, t1, t2))

print("Erro médio quadrado com KFold:", np.sqrt(np.mean(np.absolute(erros))))

anderson_estatistico, anderson_critico, anderson_significancia = stats.anderson(yh, dist='norm')
print('Anderson-Darling Statistic:', anderson_estatistico)
print('Critical Values:', anderson_critico)
print('Significance Levels:', anderson_significancia)

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(c1, c2, c3, color='blue', label='Dados reais')
ax.set_xlabel('Número de Cursos')
ax.set_ylabel('Tempo de Estudo')
ax.set_zlabel('Nota')
ax.set_title('Regressão Linear Múltipla 3D')
ax.plot_trisurf(c1, c2, yh, color='red', alpha=0.5, label='Plano de Regressão')
plt.show()