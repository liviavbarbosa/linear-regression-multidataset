import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro, anderson, kstest, norm
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat


def carregar_dados(caminho_arquivo, variaveis_independentes, variavel_dependente):
    """
    Carrega dados de um arquivo .csv ou .sav, tratando valores categóricos e ausentes.

    Args:
        caminho_arquivo (str): Caminho do arquivo de dados.
        variaveis_independentes (list): Lista das variáveis independentes.
        variavel_dependente (str): Nome da variável dependente.

    Returns:
        tuple: Arrays X (variáveis independentes), y (dependente), e DataFrame original tratado.
    """
    if caminho_arquivo.endswith('.csv'):
        df = pd.read_csv(caminho_arquivo)
    elif caminho_arquivo.endswith('.sav'):
        df, meta = pyreadstat.read_sav(caminho_arquivo)
    else:
        raise ValueError("Tipo de arquivo não suportado. Utilize arquivos .csv ou .sav")

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]

    df = df.dropna(subset=[variavel_dependente] + variaveis_independentes)

    X = df[variaveis_independentes].values
    y = df[variavel_dependente].values

    return X, y, df


def verificar_gaussianidade(y):
    """
    Realiza testes de normalidade (Shapiro-Wilk, Anderson-Darling e Kolmogorov-Smirnov)
    sobre os dados da variável dependente.

    Args:
        y (array-like): Vetor com os valores da variável dependente.
    """
    # Shapiro-Wilk
    stat_shapiro, p_shapiro = shapiro(y)
    print(f"\nShapiro-Wilk: estatística={stat_shapiro:.4f}, p-valor={p_shapiro:.4f}")
    print("Distribuição (Shapiro):", "Normal" if p_shapiro > 0.05 else "Não normal")

    # Anderson-Darling
    result_anderson = anderson(y, dist='norm')
    print(f"\nAnderson-Darling: estatística={result_anderson.statistic:.4f}")
    for i in range(len(result_anderson.critical_values)):
        sig = result_anderson.significance_level[i]
        crit = result_anderson.critical_values[i]
        result = "Rejeita normalidade" if result_anderson.statistic > crit else "Aceita normalidade"
        print(f"  Nível {sig:.1f}%: valor crítico = {crit:.4f} → {result}")

    # Kolmogorov-Smirnov
    y_norm = (y - y.mean()) / y.std()
    stat_ks, p_ks = kstest(y_norm, 'norm')
    print(f"\nKolmogorov-Smirnov: estatística={stat_ks:.4f}, p-valor={p_ks:.4f}")
    print("Distribuição (K-S):", "Normal" if p_ks > 0.05 else "Não normal")


def regressao_manual(X, y):
    """
    Realiza Regressão Linear Múltipla manualmente, utilizando a pseudo-inversa de Moore-Penrose.

    Essa função estima os coeficientes da regressão linear (beta) sem o uso de bibliotecas especializadas 
    de machine learning. Adiciona automaticamente o termo de interceptação (bias) à matriz de entrada.

    Args:
        X (array): Matriz 2D com as variáveis independentes (amostras x atributos).
        y (array): Vetor 1D com os valores da variável dependente.

    Returns:
        tuple:
            - beta (array): Vetor de coeficientes estimados da regressão (inclui o intercepto).
            - y_pred (array): Valores preditos pela regressão para os dados de entrada X.
    """
    X_ = np.hstack([np.ones((X.shape[0], 1)), X])
    beta = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
    y_pred = X_ @ beta
    return beta, y_pred


def regressao_sklearn(X, y):
    """
    Realiza regressão linear múltipla utilizando o Scikit-Learn.

    Args:
        X (array): Matriz de variáveis independentes.
        y (array): Vetor da variável dependente.

    Returns:
        tuple: Modelo treinado e predições (y_pred).
    """
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"Intercepto: {model.intercept_:.4f}")
    print(f"Coeficientes: {model.coef_}")
    return model, y_pred


def avaliar_modelo(y_true, y_pred):
    """
    Avalia o desempenho de um modelo de regressão com MSE e R².

    Args:
        y_true (array): Valores reais.
        y_pred (array): Valores preditos pelo modelo.

    Returns:
        tuple: Métricas MSE e R².
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


def validacao_k_fold(X, y, k=5):
    """
    Realiza validação cruzada k-fold com regressão linear manual.

    Args:
        X (array): Matriz de variáveis independentes.
        y (array): Vetor da variável dependente.
        k (int): Número de folds.

    Returns:
        tuple: Listas de MSE e R² por fold, além das médias de cada métrica.
    """
    n = len(X)
    fold_size = n // k
    np.random.seed(42)
    indices = np.random.permutation(n)

    mse_list = []
    r2_list = []

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size if i < k - 1 else n
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_ = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_test_ = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

        beta = np.linalg.pinv(X_train_.T @ X_train_) @ X_train_.T @ y_train
        y_pred = X_test_ @ beta

        mse, r2 = avaliar_modelo(y_test, y_pred)
        mse_list.append(mse)
        r2_list.append(r2)

    return mse_list, r2_list, np.mean(mse_list), np.mean(r2_list)


def plotar_resultados(X, y_real, y_pred):
    """
    Plota uma visualização 3D dos dados reais e do plano de regressão, além de exibir um histograma 
    dos resíduos (diferença entre valores reais e predições).

    A visualização 3D mostra as variáveis independentes (as duas primeiras colunas de X) no eixo X e Y, 
    com a variável dependente no eixo Z, e o plano de regressão que melhor se ajusta aos dados. 
    O histograma dos resíduos permite analisar a distribuição dos erros do modelo.

    Args:
        X (array): Matriz 2D contendo as variáveis independentes (deve ter pelo menos 2 colunas).
        y_real (array): Vetor 1D com os valores reais da variável dependente.
        y_pred (array): Vetor 1D com os valores preditos pelo modelo de regressão.
    """
    x1 = np.array([x[0] for x in X])
    x2 = np.array([x[1] for x in X])
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    if len(set(x1)) < 3 or len(set(x2)) < 3:
        print("Dados insuficientes para plotar em 3D (varie mais as primeiras duas features).")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, x2, y_real, c='blue', label='Valores reais')

    ax.plot_trisurf(x1, x2, y_pred, color='red', alpha=0.5, label='Plano de Regressão')

    ax.set_xlabel("Variável 1 (X₁)")
    ax.set_ylabel("Variável 2 (X₂)")
    ax.set_zlabel("Variável Dependente (Y)")
    ax.set_title("Regressão Linear Múltipla - Visualização 3D")
    plt.legend()
    plt.tight_layout()
    plt.show()

    residuos = y_real - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuos, kde=True, bins=30, color='purple')
    plt.title("Distribuição dos Resíduos")
    plt.xlabel("Erro")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def imprimir_resultados(mse_list, r2_list, mse_avg, r2_avg):
    """
    Imprime os resultados da validação cruzada k-fold.

    Args:
        mse_list (list): Lista de valores MSE por fold.
        r2_list (list): Lista de valores R² por fold.
        mse_avg (float): Média dos valores de MSE.
        r2_avg (float): Média dos valores de R².
    """
    print("\nValidação K-Fold:")
    print(f"R² médio: {r2_avg:.4f}")
    print(f"MSE médio: {mse_avg:.4f}")
    print(f"R² por fold: {np.round(r2_list, 4)}")
    print(f"MSE por fold: {np.round(mse_list, 4)}")


def executar_analise(caminho_arquivo, variaveis_independentes, variavel_dependente, k=5):
    """
    Executa o pipeline completo de análise de regressão linear.

    Args:
        caminho_arquivo (str): Caminho do arquivo de dados (.csv ou .sav).
        variaveis_independentes (list): Lista de nomes das variáveis independentes.
        variavel_dependente (str): Nome da variável dependente.
        k (int): Número de folds para validação cruzada.
    """
    nome_arquivo = os.path.basename(caminho_arquivo)
    print(f"\n===== Analisando arquivo: {nome_arquivo} =====")
    
    X, y, df = carregar_dados(caminho_arquivo, variaveis_independentes, variavel_dependente)

    print("\nVerificando distribuição da variável dependente:")
    verificar_gaussianidade(y)

    print("\n--- Regressão Linear Manual ---")
    beta, y_pred_manual = regressao_manual(X, y)
    mse_manual, r2_manual = avaliar_modelo(y, y_pred_manual)
    print(f"MSE: {mse_manual:.4f}, R²: {r2_manual:.4f}")

    print("\n--- Regressão Linear com Scikit-Learn ---")
    model, y_pred_sklearn = regressao_sklearn(X, y)
    mse_sklearn, r2_sklearn = avaliar_modelo(y, y_pred_sklearn)
    print(f"MSE: {mse_sklearn:.4f}, R²: {r2_sklearn:.4f}")

    print("\n--- Validação Cruzada (K-Fold) ---")
    mse_list, r2_list, mse_avg, r2_avg = validacao_k_fold(X, y, k)
    imprimir_resultados(mse_list, r2_list, mse_avg, r2_avg)

    plotar_resultados(X, y, y_pred_manual)