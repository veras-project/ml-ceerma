import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Definir o caminho base onde estão todas as pastas do novo conjunto de dados
# Substitua o caminho abaixo pelo "path" principal onde as pastas estão armazenadas
caminho_base = Path('../datasetventoinha/Experimento3')

lista_dfs = []

# 2. Iterar por todas as subpastas e arquivos dinamicamente
# O comando rglob('*.xlsx') vai buscar todos os arquivos Excel em qualquer subpasta
for arquivo in caminho_base.rglob('*.xlsx'):
    try:
        df_temp = pd.read_excel(arquivo)
        
        # Pega o nome da pasta imediatamente acima do arquivo para usar como classe de falha
        # Ex: se o caminho for "Experimento2/micro quebra/Coleta 1.xlsx", a classe será "micro quebra"
        nome_classe = arquivo.parent.name 
        
        df_temp['classificacao'] = nome_classe
        lista_dfs.append(df_temp)
        
    except Exception as e:
        print(f"Erro ao ler o arquivo {arquivo.name}: {e}")

# 3. Junta todos os DataFrames carregados em um só
df_unificado = pd.concat(lista_dfs, ignore_index=True)

# 4. Garante que Tempo_ms seja numérico e cria a janela de tempo
tamanho_janela_ms = 333

df_unificado['Tempo_ms'] = pd.to_numeric(df_unificado['Tempo_ms'], errors='coerce')
df_unificado = df_unificado.dropna(subset=['Tempo_ms'])

# Cria a coluna ID_Janela
df_unificado['ID_Janela'] = (df_unificado['Tempo_ms'] // tamanho_janela_ms).astype(int)

# 5. Extração das Features (Agrupamento por classificação e ID_Janela)
df_features = (
    df_unificado.groupby(['classificacao', 'ID_Janela'])
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
print("Dimensão do dataset final:", df_features.shape)

# 6. Preparação das variáveis X e y
X = df_features.drop(columns=['classificacao', 'ID_Janela'])
y = df_features['classificacao']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=67, stratify=y
)

# 7. Treinamento e avaliação com RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=67)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia Total: {acc * 100:.2f}%\n")
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))