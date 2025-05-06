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
    X_ = np.hstack([np.ones((X.shape[0], 1)), X])
    beta = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
    y_pred = X_ @ beta
    return beta, y_pred


def regressao_sklearn(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"Intercepto: {model.intercept_:.4f}")
    print(f"Coeficientes: {model.coef_}")
    return model, y_pred


def avaliar_modelo(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


def validacao_k_fold(X, y, k=5):
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


def plotar_resultados(y, y_pred):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=y, y=y_pred, line_kws={"color": "red"}, ci=95)
    plt.xlabel("Valores reais")
    plt.ylabel("Previsões")
    plt.title("Regressão Linear: Valores Reais vs Previsões")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotando resíduos
    residuos = y - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuos, kde=True, bins=30)
    plt.title("Distribuição dos Resíduos")
    plt.xlabel("Erro")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def imprimir_resultados(mse_list, r2_list, mse_avg, r2_avg):
    print("\nValidação K-Fold:")
    print(f"R² médio: {r2_avg:.4f}")
    print(f"MSE médio: {mse_avg:.4f}")
    print(f"R² por fold: {np.round(r2_list, 4)}")
    print(f"MSE por fold: {np.round(mse_list, 4)}")


def executar_analise(caminho_arquivo, variaveis_independentes, variavel_dependente, k=5):
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

    plotar_resultados(y, y_pred_sklearn)