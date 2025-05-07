import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def column(matrix, i):
    return [float(row[i]) for row in matrix]

dados = pd.read_csv('Student_Marks.csv', sep=',')
dados.head()

c1 = column(dados.values, 0)
c2 = column(dados.values, 1)
c3 = column(dados.values, 2)

kf = KFold(n_splits=5, shuffle=True)

regressao = LinearRegression()
X = np.array(list(zip(c1, c2)))
y = np.array(c3)

regressao.fit(X, y)
print('R²:', regressao.score(X, y))
print("Valor Previsão:", regressao.predict([[1, 2]]))

pontos = cross_val_score(regressao, X, y, cv=kf, scoring='neg_mean_squared_error')
print("Erro médio quadrado com KFold:", np.sqrt(np.mean(np.absolute(pontos))))

anderson_estatistico, anderson_critico, anderson_significancia = stats.anderson(regressao.predict(X), dist='norm')
print('Teste Anderson-Darling:', anderson_estatistico)
print('Valor crítico:', anderson_critico)
print('Nível de significancia:', anderson_significancia)

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(c1, c2, c3, color='blue', label='Dados reais')
ax.set_xlabel('Número de Cursos')
ax.set_ylabel('Tempo de Estudo')
ax.set_zlabel('Nota')
ax.set_title('Regressão Linear Múltipla 3D')
ax.plot_trisurf(c1, c2, regressao.predict(X), color='red', alpha=0.5, label='Plano de Regressão')
plt.show()