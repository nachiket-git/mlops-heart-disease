import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Column names (14 cols; from heart-disease.names)
# -----------------------------
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]


# -----------------------------
# Load data, mark -9.0 as missing
# -----------------------------
df = pd.read_csv(
    './data/processed.cleveland.data',
    names=columns,
    na_values=['-9.0', '?']
)


# -----------------------------
# Show first few rows
# -----------------------------
print(df.head())


# =============================
# Summary statistics
# =============================
desc = df.describe()
print(desc)
desc.to_csv('./figures/desc_summary.csv')


# =============================
# Missing value analysis
# =============================
missing = df.isnull().sum()
print("Missing values:\n", missing)

missing.plot(kind='bar')
plt.title('Missing Values per Feature')
plt.tight_layout()
plt.savefig('./figures/missing_values.png')
plt.close()


# =============================
# Histograms per feature
# =============================
for col in columns[:-1]:
    plt.figure()
    df[col].hist(bins=20)
    plt.title(f'Histogram: {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'./figures/hist_{col}.png')
    plt.close()


# =============================
# Correlation heatmap
# =============================
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('./figures/corr_heatmap.png')
plt.close()


# =============================
# Class imbalance / barplot (target)
# =============================
plt.figure()
df['target'].value_counts().sort_index().plot(kind='bar')
plt.title('Class Balance (target)')
plt.xlabel('target')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('./figures/class_balance.png')
plt.close()


print("EDA Complete. Figures saved in figures/")
