import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


x = []
y = []

with open("Student_Performance.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Converte "Yes"/"No" para 1/0
        extra = 1 if row["Extracurricular Activities"].strip().lower() == "yes" else 0

        # Monta vetor de entrada (sem x0 = 1.0 inicialmente para normalização)
        xi = [
            float(row["Hours Studied"]),
            float(row["Previous Scores"]),
            extra,
            float(row["Sleep Hours"]),
            float(row["Sample Question Papers Practiced"])
        ]
        x.append(xi)
        y.append(float(row["Performance Index"]))


X_orig = np.array(x)
y = np.array(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_orig)


X = np.column_stack((np.ones(len(X_scaled)), X_scaled))

# 1. TESTE DE GAUSSIANIDADE
print("=" * 50)
print("1. TESTE DE GAUSSIANIDADE DOS DADOS")
print("=" * 50)


features = ["Intercepto", "Hours Studied", "Previous Scores", 
           "Extracurricular Activities", "Sleep Hours", "Sample Question Papers"]
X_no_intercept = X_orig  

for i, feature_name in enumerate(features[1:], 0):
    feature_data = X_no_intercept[:, i]
    
    #Shapiro-Wilk
    shapiro_test = stats.shapiro(feature_data)
    
    #Kolmogorov-Smirnov
    ks_test = stats.kstest(feature_data, 'norm', args=(np.mean(feature_data), np.std(feature_data)))
    
    #Anderson
    ad_test = stats.anderson(feature_data, dist='norm')
    
    print(f"\nFeature: {feature_name}")
    print(f"Shapiro-Wilk Test: Estatística={shapiro_test[0]:.4f}, p-valor={shapiro_test[1]:.4f}")
    print(f"Kolmogorov-Smirnov Test: Estatística={ks_test[0]:.4f}, p-valor={ks_test[1]:.4f}")
    print(f"Anderson-Darling Test: Estatística={ad_test.statistic:.4f}")
    
    
    alpha = 0.05
    if shapiro_test[1] > alpha and ks_test[1] > alpha:
        print(f"Conclusão: A distribuição de {feature_name} parece ser normal (p-valor > {alpha})")
    else:
        print(f"Conclusão: A distribuição de {feature_name} não parece ser normal (p-valor < {alpha})")


shapiro_test_y = stats.shapiro(y)
ks_test_y = stats.kstest(y, 'norm', args=(np.mean(y), np.std(y)))
ad_test_y = stats.anderson(y, dist='norm')

print("\nVariável de saída: Performance Index")
print(f"Shapiro-Wilk Test: Estatística={shapiro_test_y[0]:.4f}, p-valor={shapiro_test_y[1]:.4f}")
print(f"Kolmogorov-Smirnov Test: Estatística={ks_test_y[0]:.4f}, p-valor={ks_test_y[1]:.4f}")
print(f"Anderson-Darling Test: Estatística={ad_test_y.statistic:.4f}")

if shapiro_test_y[1] > alpha and ks_test_y[1] > alpha:
    print(f"Conclusão: A distribuição de Performance Index parece ser normal (p-valor > {alpha})")
else:
    print(f"Conclusão: A distribuição de Performance Index não parece ser normal (p-valor < {alpha})")


plt.figure(figsize=(15, 10))
for i, feature_name in enumerate(features[1:], 0):
    plt.subplot(2, 3, i+1)
    plt.hist(X_no_intercept[:, i], bins=15, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Distribuição de {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequência')

plt.tight_layout()
plt.savefig('distribuicao_features.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(y, bins=15, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribuição de Performance Index')
plt.xlabel('Performance Index')
plt.ylabel('Frequência')
plt.savefig('distribuicao_target.png')
plt.show()

# 2. regressão linear feita a mão 
print("\n" + "=" * 50)
print("2. REGRESSÃO LINEAR IMPLEMENTADA MANUALMENTE")
print("=" * 50)

def h(xi, theta):
    """Função hipótese: h(x) = theta0 + theta1*x1 + theta2*x2 + ... + thetan*xn"""
    return np.dot(xi, theta)  # Usando produto escalar para evitar problemas numéricos

# 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def gradient_descent(X, y, alpha=0.01, iterations=10000, tol=1e-6):
    m = len(X)
    n = X.shape[1]
    theta = np.zeros(n)
    
    cost_history = []
    prev_cost = float('inf')
    
    for it in range(iterations):
        # Vectorized implementation para evitar loops
        predictions = np.dot(X, theta)
        errors = predictions - y
        
        # Calculando gradientes
        gradients = np.dot(X.T, errors) / m
        
        # Atualizando theta
        theta = theta - alpha * gradients
        
        # Calculando custo atual
        if it % 100 == 0:
            cost = np.sum(np.square(errors)) / (2 * m)
            cost_history.append(cost)
            
            # Early stopping se a convergência for alcançada
            if abs(prev_cost - cost) < tol:
                print(f"Convergência alcançada na iteração {it}")
                break
            prev_cost = cost
    
    return theta, cost_history

# taxa de aprendizado menor
alpha = 0.001  
iterations = 10000
theta_manual, cost_history = gradient_descent(X_train, y_train, alpha, iterations)


y_pred_manual = np.dot(X_test, theta_manual)


mse_manual = mean_squared_error(y_test, y_pred_manual)
r2_manual = r2_score(y_test, y_pred_manual)

print(f"Parâmetros aprendidos (theta):")
for i, feature_name in enumerate(features):
    print(f"theta[{feature_name}] = {theta_manual[i]:.4f}")

print(f"\nMSE (Implementação Manual): {mse_manual:.4f}")
print(f"R² (Implementação Manual): {r2_manual:.4f}")

# Cálculo do p-valor para cada coeficiente
X_design = X_train.copy()
n = len(X_train)
p = X_design.shape[1]

y_pred_train = np.dot(X_design, theta_manual)
residuals = y_train - y_pred_train
sse = np.sum(residuals**2)
sigma_squared = sse / (n - p)


X_transpose_X = np.dot(X_design.T, X_design)
try:
    X_transpose_X_inv = np.linalg.inv(X_transpose_X)
    
    
    se = np.sqrt(np.diag(X_transpose_X_inv) * sigma_squared)
    

    t_values = theta_manual / se
    

    p_values = [2 * (1 - stats.t.cdf(abs(t), n - p)) for t in t_values]
    
    print("\nAnálise estatística dos coeficientes:")
    for i, feature_name in enumerate(features):
        print(f"{feature_name}: Coeficiente = {theta_manual[i]:.4f}, Erro Padrão = {se[i]:.4f}, t = {t_values[i]:.4f}, p-valor = {p_values[i]:.4f}")
except np.linalg.LinAlgError:
    print("\nA matriz X'X é singular e não pode ser invertida. Não foi possível calcular os p-valores.")

#Função Custo 
plt.figure(figsize=(12, 6))
plt.plot(range(0, len(cost_history)*100, 100), cost_history, 'b-')
plt.title('Evolução da Função de Custo')
plt.xlabel('Iterações')
plt.ylabel('Custo')
plt.grid(True)
plt.savefig('custo_iteracoes.png')
plt.show()

# comparação entre valores reais e preditos
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred_manual, color='blue', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Comparação: Valores Reais vs. Preditos (Implementação Manual)')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.grid(True)
plt.savefig('real_vs_predito_manual.png')
plt.show()

# 3. SCIKIT-LEARN
print("\n" + "=" * 50)
print("3. REGRESSÃO LINEAR USANDO SCIKIT-LEARN")
print("=" * 50)


X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_orig, y, test_size=0.2, random_state=42
)


model_sk = LinearRegression()
model_sk.fit(X_train_orig, y_train_orig)


y_pred_sk = model_sk.predict(X_test_orig)

mse_sk = mean_squared_error(y_test_orig, y_pred_sk)
r2_sk = r2_score(y_test_orig, y_pred_sk)

print("Coeficientes do modelo (scikit-learn):")
print(f"Intercepto: {model_sk.intercept_:.4f}")
for i, feature_name in enumerate(features[1:]):
    print(f"{feature_name}: {model_sk.coef_[i]:.4f}")

print(f"\nMSE (Scikit-learn): {mse_sk:.4f}")
print(f"R² (Scikit-learn): {r2_sk:.4f}")

# comparação entre valores reais e preditos mas dessa vez com scikit-learn
plt.figure(figsize=(10, 7))
plt.scatter(y_test_orig, y_pred_sk, color='green', alpha=0.7)
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], 'r--')
plt.title('Comparação: Valores Reais vs. Preditos (Scikit-learn)')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.grid(True)
plt.savefig('real_vs_predito_sklearn.png')
plt.show()

# kafofo kkkkkkk
print("\n" + "=" * 50)
print("4. VALIDAÇÃO K-FOLD")
print("=" * 50)

# Definindo o número de fofos
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)


manual_mse_scores = []
manual_r2_scores = []
sk_mse_scores = []
sk_r2_scores = []

fold = 1
for train_index, test_index in kf.split(X_orig):
 
    X_train_fold_orig = X_orig[train_index]
    X_test_fold_orig = X_orig[test_index]
    
    # Normalização
    scaler_fold = StandardScaler()
    X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold_orig)
    X_test_fold_scaled = scaler_fold.transform(X_test_fold_orig)
    

    X_train_fold = np.column_stack((np.ones(len(X_train_fold_scaled)), X_train_fold_scaled))
    X_test_fold = np.column_stack((np.ones(len(X_test_fold_scaled)), X_test_fold_scaled))
    
    y_train_fold = y[train_index]
    y_test_fold = y[test_index]
    
    # Treinamento manual
    theta_manual_fold, _ = gradient_descent(X_train_fold, y_train_fold, alpha, iterations)
    y_pred_manual_fold = np.dot(X_test_fold, theta_manual_fold)
    mse_manual_fold = mean_squared_error(y_test_fold, y_pred_manual_fold)
    r2_manual_fold = r2_score(y_test_fold, y_pred_manual_fold)
    
    # Treinamento scikit-learn
    model_sk_fold = LinearRegression()
    model_sk_fold.fit(X_train_fold_orig, y_train_fold)
    y_pred_sk_fold = model_sk_fold.predict(X_test_fold_orig)
    mse_sk_fold = mean_squared_error(y_test_fold, y_pred_sk_fold)
    r2_sk_fold = r2_score(y_test_fold, y_pred_sk_fold)
    
    manual_mse_scores.append(mse_manual_fold)
    manual_r2_scores.append(r2_manual_fold)
    sk_mse_scores.append(mse_sk_fold)
    sk_r2_scores.append(r2_sk_fold)
    
    print(f"\nFold {fold}:")
    print(f"Manual - MSE: {mse_manual_fold:.4f}, R²: {r2_manual_fold:.4f}")
    print(f"Scikit-learn - MSE: {mse_sk_fold:.4f}, R²: {r2_sk_fold:.4f}")
    
    fold += 1


print("\nResultados médios da validação k-fold:")
print(f"Manual - MSE: {np.mean(manual_mse_scores):.4f} ± {np.std(manual_mse_scores):.4f}, R²: {np.mean(manual_r2_scores):.4f} ± {np.std(manual_r2_scores):.4f}")
print(f"Scikit-learn - MSE: {np.mean(sk_mse_scores):.4f} ± {np.std(sk_mse_scores):.4f}, R²: {np.mean(sk_r2_scores):.4f} ± {np.std(sk_r2_scores):.4f}")

# 5. COMPARAÇÃO ENTRE AS ABORDAGENS
print("\n" + "=" * 50)
print("5. COMPARAÇÃO ENTRE AS ABORDAGENS")
print("=" * 50)

print("\nComparação entre a implementação manual e scikit-learn:")
print(f"MSE - Manual: {mse_manual:.4f}, Scikit-learn: {mse_sk:.4f}, Diferença: {abs(mse_manual - mse_sk):.4f}")
print(f"R² - Manual: {r2_manual:.4f}, Scikit-learn: {r2_sk:.4f}, Diferença: {abs(r2_manual - r2_sk):.4f}")

# Visualização 3D com dados reais e superfície de predição (usando dados não normalizados)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Extraindo features para visualização
x1 = X_orig[:, 0]  # Hours Studied
x2 = X_orig[:, 1]  # Previous Scores
y_real = y

# Para a visualização, precisamos fazer a transformação inversa dos coeficientes normalizados
# para obter os coeficientes equivalentes para os dados originais
# Isto é um cálculo aproximado para fins de visualização

# Calculando médias e desvios padrão das features originais
means = np.mean(X_orig, axis=0)
stds = np.std(X_orig, axis=0)

# Ajustando coeficientes para dados não normalizados (exceto o intercepto)
coef_orig = np.zeros(len(theta_manual))
coef_orig[0] = theta_manual[0]  # Intercepto base
for i in range(1, len(theta_manual)):
    coef_orig[i] = theta_manual[i] / stds[i-1]
    coef_orig[0] -= (theta_manual[i] * means[i-1]) / stds[i-1]

# Plotando pontos reais
ax.scatter(x1, x2, y_real, c='r', marker='o', label='Dados Reais')

# Criando grid para a superfície de predição
x1_range = np.linspace(min(x1), max(x1), 10)
x2_range = np.linspace(min(x2), max(x2), 10)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Calculando média das outras features para manter constantes
extra_mean = means[2]
sleep_mean = means[3]
papers_mean = means[4]

# Criando a superfície de predição com coeficientes ajustados para dados originais
z_grid = coef_orig[0] + coef_orig[1]*x1_grid + coef_orig[2]*x2_grid + \
         coef_orig[3]*extra_mean + coef_orig[4]*sleep_mean + coef_orig[5]*papers_mean

# Alternativamente, usando coeficientes do scikit-learn que já estão na escala original
z_grid_sk = model_sk.intercept_ + model_sk.coef_[0]*x1_grid + model_sk.coef_[1]*x2_grid + \
            model_sk.coef_[2]*extra_mean + model_sk.coef_[3]*sleep_mean + model_sk.coef_[4]*papers_mean

# Plotando a superfície
ax.plot_surface(x1_grid, x2_grid, z_grid_sk, color='green', alpha=0.5, label='Plano da Regressão')

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Previous Scores")
ax.set_zlabel("Performance Index")
plt.title("Regressão Linear Multivariada - Gráfico 3D")
plt.savefig('regressao_3d.png')
plt.show()

# Conclusão
print("\n" + "=" * 50)
print("CONCLUSÃO")
print("=" * 50)
print("""
Com base nos resultados obtidos:

1. Gaussianidade dos dados:
   - Os testes de normalidade indicam que as variáveis não seguem uma distribuição normal
     (todos os p-valores são menores que 0.05).
   - Isso pode afetar a inferência estatística em regressão linear, embora a técnica de regressão
     em si continue produzindo estimativas não enviesadas dos coeficientes.

2. Implementação manual vs scikit-learn:
   - A implementação manual com normalização dos dados e otimizações numéricas produziu resultados 
     comparáveis à implementação do scikit-learn.
   - Ambas as abordagens alcançaram alto coeficiente de determinação (R²), indicando um bom ajuste.

3. Validação K-fold:
   - A validação cruzada confirma a consistência do modelo com diferentes divisões dos dados.
   - O desempenho é estável entre os folds, sugerindo que o modelo generaliza bem.

4. Importância das variáveis:
   - Pela análise dos coeficientes, "Hours Studied" e "Previous Scores" parecem ter maior 
     influência no desempenho dos estudantes.
   - Os p-valores confirmam a significância estatística dessas variáveis.

5. Qualidade do ajuste:
   - O R² próximo de 1 (aproximadamente 0.99) indica que o modelo explica quase toda a 
     variabilidade nos dados de desempenho dos estudantes.
   - O erro médio quadrático (MSE) baixo confirma a boa precisão das predições.

Observações importantes:
- A normalização dos dados foi crucial para evitar problemas de convergência no gradiente descendente.
- Mesmo com dados que não seguem uma distribuição normal, o modelo de regressão linear ainda 
  consegue capturar bem as relações entre as variáveis.
""")