import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Load the dataset
df = pd.read_csv('cic_2023.csv')
print("All the attack")

# Check if 'Unnamed: 0' exists in the columns before dropping
if 'Unnamed: 0' in df.columns:
    df_data = df.drop(['Unnamed: 0'], axis=1)
else:
    df_data = df  # If 'Unnamed: 0' does not exist, keep the original DataFrame

# Display the first few rows
print(df_data.head())

# Display the attack types
print("==Attack Type==")
print(df['label'].value_counts())