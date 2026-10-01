import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df_semfalha = pd.read_excel('datasetventoinha/Experimento2/1 - Sem falhas/Coleta 1.xlsx')

df_semfalha['classificacao'] = 'sem falha'

df_semfalha.info()
df_semfalha.head()