# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the Dataset
df = pd.read_csv('C:/Users/pkadi/OneDrive/Documents/titanic_dataset_cleaning/Titanic-Dataset.csv')

# Step 3: Explore the Dataset
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.dtypes)

# Step 4: Handle Missing Values
# Fill 'Age' with median, 'Embarked' with mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Step 5: Encode Categorical Features
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])         # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])  # S=2, C=0, Q=1

# Step 6: Normalize Numerical Features
scaler = StandardScaler()
num_features = ['Age', 'Fare']
df[num_features] = scaler.fit_transform(df[num_features])

# Step 7: Visualize Outliers
for col in num_features:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Step 8: Remove Outliers using IQR method
for col in num_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# Final check
print("After cleaning:")
print(df.info())
