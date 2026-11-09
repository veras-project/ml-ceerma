import pandas as pd
import numpy as np
from pathlib import Path
import re
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# 1. Configurações Iniciais
CAMINHO_BASE = Path('../content/drive/MyDrive/Temp')
TAMANHO_JANELA_MS = 333
RANDOM_SEED = 67

lista_dfs = []

# 2. Iteração e Limpeza
for id_coleta, ficheiro in enumerate(CAMINHO_BASE.rglob('*.xlsx')):
    try:
        df_temp = pd.read_excel(ficheiro)

        nome_pasta = ficheiro.parent.name
        nome_arquivo = ficheiro.stem

        if "coleta" in nome_arquivo.lower():
            nome_bruto = nome_pasta
        else:
            nome_bruto = nome_arquivo

        # Limpeza para o Machine Learning (remove números e hifens)
        nome_limpo = re.sub(r'^\d+\s*-\s*', '', nome_bruto)
        nome_classe_ml = nome_limpo.strip().capitalize()

        # Guarda ambas as informações
        df_temp['nome_original'] = nome_bruto      # Para a sua tabela (ex: 1 - Sem falhas)
        df_temp['classificacao'] = nome_classe_ml  # Para o modelo (ex: Sem falhas)
        df_temp['id_coleta'] = id_coleta

        lista_dfs.append(df_temp)

    except Exception as e:
        print(f"Erro ao ler o ficheiro {ficheiro.name}: {e}")

if not lista_dfs:
    raise ValueError("Nenhum ficheiro Excel foi encontrado. Verifique o caminho base.")

# 3. Junta tudo
df_unificado = pd.concat(lista_dfs, ignore_index=True)

# 4. Formatação de Tempo
df_unificado['Tempo_ms'] = pd.to_numeric(df_unificado['Tempo_ms'], errors='coerce')
df_unificado = df_unificado.dropna(subset=['Tempo_ms'])
df_unificado['ID_Janela'] = (df_unificado['Tempo_ms'] // TAMANHO_JANELA_MS).astype(int)

# 5. Extração das Features
# Adicionamos 'nome_original' no groupby para não o perder
df_features = (
    df_unificado.groupby(['classificacao', 'nome_original', 'id_coleta', 'ID_Janela'])
    .agg(
        Ax_Mean=('Ax_g', 'mean'),
        Ax_Std=('Ax_g', 'std'),
        Ax_Max=('Ax_g', 'max'),
        Ax_Min=('Ax_g', 'min'),
        Ay_Mean=('Ay_g', 'mean'),
        Ay_Std=('Ay_g', 'std'),
        Ay_Max=('Ay_g', 'max'),
        Ay_Min=('Ay_g', 'min'),
        Az_Mean=('Az_g', 'mean'),
        Az_Std=('Az_g', 'std'),
        Az_Max=('Az_g', 'max'),
        Az_Min=('Az_g', 'min'),
    )
    .reset_index()
)

df_features = df_features.dropna().reset_index(drop=True)

# 6. Preparação para o Machine Learning
# Removemos todas as colunas de texto/identificação do X
X = df_features.drop(columns=['classificacao', 'nome_original', 'id_coleta', 'ID_Janela'])
# O Y recebe apenas a classificação limpa, avaliando as origens diferentes como uma classe só
y = df_features['classificacao']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

# 7. Treino e Avaliação
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia Total: {acc * 100:.2f}%\n")
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. GRÁFICO DE DISPERSÃO (DATA DISPOSITION)
# ==========================================
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Usamos 'nome_original' no hue para você ver cada coleta/posição com uma cor única.
# A paleta 'tab20' suporta até 20 cores diferentes, ideal para muitos ficheiros.
sns.scatterplot(
    data=df_features,
    x='Ax_Std',
    y='Ay_Std',
    hue='nome_original',
    palette='tab20',
    alpha=0.7,
    s=50
)

plt.title('Dispersão das Vibrações por Coleta (Eixo X vs Eixo Y)', fontsize=14, pad=15)
plt.xlabel('Vibração no Eixo X (Desvio Padrão)', fontsize=12)
plt.ylabel('Vibração no Eixo Y (Desvio Padrão)', fontsize=12)

# Coloca a legenda fora do gráfico para não tampar os dados
plt.legend(title='Origem do Dado', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()


# ==========================================
# 2. MATRIZ DE CONFUSÃO (CONFUSION MATRIX)
# ==========================================
plt.figure(figsize=(10, 8))

# Calcula a matriz usando as variáveis de teste e previsão geradas anteriormente
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

# Cria o mapa de calor (heatmap) para a matriz
sns.heatmap(
    cm,
    annot=True,        # Mostra os números dentro dos quadrados
    fmt='d',           # Formata como número inteiro
    cmap='Blues',      # Esquema de cores (azul facilita a leitura)
    xticklabels=rf_model.classes_,
    yticklabels=rf_model.classes_
)

plt.title('Matriz de Confusão - Random Forest', fontsize=14, pad=15)
plt.xlabel('Previsão do Modelo (O que a IA achou que era)', fontsize=12)
plt.ylabel('Classe Real (O que realmente era)', fontsize=12)
plt.xticks(rotation=45, ha='right') # Inclina os nomes embaixo para não encavalarem
plt.tight_layout()
plt.show()