# analysis.py

# -----------------------------------
# 1. Import Libraries (Data Wrangling I)
# -----------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# 2. Problem Definition
# -----------------------------------
print("Problem: Predict Spotify song popularity using audio features")

# -----------------------------------
# 3. Data Collection
# -----------------------------------
df = pd.read_csv("spotify.csv")
print("\nDataset loaded successfully")

# -----------------------------------
# 4. Data Understanding
# -----------------------------------
print("\nShape of dataset:", df.shape)
print("\nColumns:\n", df.columns)
print("\nData Info:")
df.info()

# -----------------------------------
# 5. Missing Values Check
# -----------------------------------
print("\nMissing Values:\n", df.isnull().sum())

# -----------------------------------
# 6. Data Formatting
# -----------------------------------
print("\nData Types:\n", df.dtypes)

# -----------------------------------
# 7. Feature Selection (Data Reduction)
# -----------------------------------
features = [
    'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness',
    'instrumentalness', 'liveness',
    'valence', 'tempo', 'popularity'
]

df = df[features]

# -----------------------------------
# 8. Data Cleaning
# -----------------------------------
df = df.dropna()
print("\nData after cleaning:", df.shape)

# -----------------------------------
# 9. Descriptive Statistics
# -----------------------------------
print("\nStatistical Summary:\n", df.describe())

# -----------------------------------
# 10. Outlier Detection
# -----------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=df)
plt.title("Boxplot for Outlier Detection")
plt.xticks(rotation=45)
plt.show()

# -----------------------------------
# 11. EDA - Distribution
# -----------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['popularity'], bins=30)
plt.title("Popularity Distribution")
plt.show()

# -----------------------------------
# 12. Feature Relationships
# -----------------------------------
sns.scatterplot(x='energy', y='popularity', data=df)
plt.title("Energy vs Popularity")
plt.show()

sns.scatterplot(x='danceability', y='popularity', data=df)
plt.title("Danceability vs Popularity")
plt.show()

# -----------------------------------
# 13. Correlation Analysis
# -----------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------------
# 14. Insights
# -----------------------------------
print("\nInsights:")
print("1. Energy and danceability have a positive influence on popularity.")
print("2. No single feature strongly dominates popularity.")
print("3. Dataset is clean after removing missing values.")