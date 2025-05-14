# iris_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Task 1: Load and Explore Dataset
# -------------------------------

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check structure
print("\nDataset info:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean dataset (drop missing values if any)
df = df.dropna()

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

# Basic statistics
print("\nDescriptive statistics:")
print(df.describe())

# Grouping by species and computing mean petal length
grouped = df.groupby('species')['petal_length'].mean()
print("\nAverage petal length by species:")
print(grouped)

# Find species with the highest average petal length
print("\nSpecies with highest average petal length:", grouped.idxmax())

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

# Set style
sns.set(style="whitegrid")

# 1. Line Chart (simulated trend with cumulative mean)
df_sorted = df.sort_values('sepal_length')
df_sorted['cumulative_mean'] = df_sorted['sepal_length'].expanding().mean()

plt.figure(figsize=(8, 4))
plt.plot(df_sorted.index, df_sorted['cumulative_mean'], label='Cumulative Mean Sepal Length')
plt.title('Trend of Cumulative Sepal Length')
plt.xlabel('Index')
plt.ylabel('Cumulative Mean')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart - Average Petal Length per Species
plt.figure(figsize=(8, 4))
sns.barplot(x='species', y='petal_length', data=df, estimator='mean', ci=None)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length')
plt.tight_layout()
plt.show()

# 3. Histogram - Sepal Width
plt.figure(figsize=(8, 4))
plt.hist(df['sepal_width'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(8, 4))
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(title='Species')
plt.tight_layout()
plt.show()
