import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def extract_vibration_features(signal, window_size=1000, step=500):
    """Extrai características nos domínios do tempo e da frequência."""
    features = []

    for i in range(0, len(signal) - window_size + 1, step):
        win = signal[i : i + window_size]

        # Domínio do Tempo
        rms = np.sqrt(np.mean(win**2))  # Energia média do sinal
        std = np.std(win)  # Dispersão
        p2p = np.max(win) - np.min(win)  # Amplitude Pico a Pico
        kurt = kurtosis(win)  # Sensível a impactos/falhas pontuais
        skw = skew(win)  # Assimetria da distribuição

        # Domínio da Frequência (FFT)
        fft_vals = np.abs(np.fft.rfft(win))
        spectral_energy = np.sum(fft_vals**2)  # Energia espectral total
        dom_freq_idx = np.argmax(
            fft_vals[1:]
        ) + 1  # Frequência de maior amplitude

        features.append(
            [rms, std, p2p, kurt, skw, spectral_energy, dom_freq_idx]
        )

    cols = [
        "RMS",
        "STD",
        "P2P",
        "Kurtosis",
        "Skewness",
        "Energy",
        "DomFreqIdx",
    ]
    return pd.DataFrame(features, columns=cols)


def process_dataset(filepath, label, axis_col="acc_z"):
    """Lê o arquivo, calcula a aceleração resultante (se necessário) e extrai features."""
    df = pd.read_csv(filepath)

    # Caso possua os 3 eixos (X, Y, Z), calcula o vetor resultante
    if set(["acc_x", "acc_y", "acc_z"]).issubset(df.columns):
        signal = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    else:
        signal = df[axis_col].values

    features_df = extract_vibration_features(signal)
    features_df["target"] = label
    return features_df


# --- 1. Carga e Pré-processamento dos 3 Conjuntos de Dados ---
# Substitua pelos caminhos dos seus arquivos CSV
df_top = process_dataset("ventoinha_top.csv", label=0)
df_quarto = process_dataset("quarto_quebra.csv", label=1)
df_total = process_dataset("quebra_total.csv", label=2)

# Unificação da base de dados
full_df = pd.concat([df_top, df_quarto, df_total], ignore_index=True)

# --- 2. Divisão de Treino e Teste ---
X = full_df.drop(columns=["target"])
y = full_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 3. Treinamento do Modelo de Classificação ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. Avaliação do Desempenho ---
y_pred = model.predict(X_test)
labels_map = ["Ventoinha Top", "1/4 de Quebra", "Quebra Total"]

print("=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, y_pred, target_names=labels_map))

print("\n=== MATRIZ DE CONFUSÃO ===")
print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=labels_map, columns=labels_map))