import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from scipy import stats

df = pd.read_csv('Sales.csv')


print(" ORIGINAL DATA INFO")
print("=" * 50)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print("=" * 50)

# =========================
# 2 Remove duplicates
# =========================
df.drop_duplicates(inplace=True)
print(f"\n Duplicates removed. New rows: {len(df)}")

# =========================
# 3 Clean Memory & Storage
# =========================
df['clean_memory'] = (df['Memory'].str.replace('GB', '', case=False).str.replace('MB', '', case=False).str.strip()) 
df['clean_storage'] = (df['Storage'].str.replace('GB', '', case=False).str.replace('MB', '', case=False).str.strip())

# Convert to numeric, coercing errors to NaN
df['clean_memory'] = pd.to_numeric(df['clean_memory'], errors='coerce')
df['clean_storage'] = pd.to_numeric(df['clean_storage'], errors='coerce')

df.drop(columns=["Memory", "Storage"], inplace=True)


# 4 Handle Missing Values

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

# Numeric columns to Median
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns to Mode (most frequent)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\n Missing values handled!")
print(f"Missing values after cleaning:\n{df.isnull().sum()}")


# STATISTICAL METHOD FOR OUTLIER DETECTION here i use (Z-SCORE)

numeric_features = ['Selling Price', 'Original Price', 'clean_memory', 'clean_storage', 'Rating']


z_scores = np.abs(stats.zscore(df[numeric_features]))


outliers = (z_scores > 3)

print("\n Outlier detection using Z-Score method:")
for col in numeric_features:
    num_outliers = outliers[:, numeric_features.index(col)].sum()
    print(f" {col}: {num_outliers} outliers detected")


#VISUALIZATION BEFORE ENCODING


#1. Top 10 Brands - Bar Chart
print("\n Creating visualizations...")
brand_counts = df["Brands"].value_counts()

plt.figure(figsize=(12, 6))
brand_counts.head(10).plot(kind="bar", color='skyblue', edgecolor='black')
plt.xlabel("Brand", fontsize=12)
plt.ylabel("Number of Phones", fontsize=12)
plt.title("Top 10 Most Frequent Brands in Dataset", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#2. Brand Distribution - Pie Chart (Top 5)
plt.figure(figsize=(10, 8))
top_5_brands = df['Brands'].value_counts().head(5)
colors = plt.cm.Set3(range(len(top_5_brands)))
plt.pie(top_5_brands, labels=top_5_brands.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title("Top 5 Brands Distribution", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#3. Best-Selling Model for Each Brand - Bar Chart
brand_model_counts = (
    df.groupby(["Brands", "Models"]).size()
    .reset_index(name="count")
)
top_model_each_brand = (
    brand_model_counts
    .sort_values(["Brands", "count"], ascending=[True, False])
    .drop_duplicates(subset="Brands", keep="first")
    .sort_values("count", ascending=False)
    .head(10)
)
top_model_each_brand['Brand_Model'] = (top_model_each_brand['Brands'] + 
                                        ' - ' + top_model_each_brand['Models'])

plt.figure(figsize=(14, 6))
plt.bar(top_model_each_brand["Brand_Model"], top_model_each_brand["count"], 
        color='coral', edgecolor='black')
plt.xlabel("Model", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Best-Selling Model for Each Brand (Top 10)", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#4. Price Distribution - Histogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['Selling Price'].dropna(), bins=30, color='lightgreen', edgecolor='black')
plt.xlabel("Selling Price (EGP)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Selling Price Distribution", fontsize=14, fontweight='bold')

plt.subplot(1, 2, 2)
plt.hist(df['Original Price'].dropna(), bins=30, color='lightcoral', edgecolor='black')
plt.xlabel("Original Price (EGP)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Original Price Distribution", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

#5. RAM vs Selling Price - Scatter Plot
df_ram_price = df.dropna(subset=["clean_memory", "Selling Price"])

plt.figure(figsize=(12, 6))
plt.scatter(df_ram_price["clean_memory"], df_ram_price["Selling Price"], 
            alpha=0.6, c='purple', edgecolors='black', s=50)
plt.xlabel("RAM (GB)", fontsize=12)
plt.ylabel("Selling Price (EGP)", fontsize=12)
plt.title("Relationship between RAM and Selling Price", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#6. Storage vs Selling Price - Scatter Plot
df_storage_price = df.dropna(subset=["clean_storage", "Selling Price"])

plt.figure(figsize=(12, 6))
plt.scatter(df_storage_price["clean_storage"], df_storage_price["Selling Price"], 
            alpha=0.6, c='orange', edgecolors='black', s=50)
plt.xlabel("Storage (GB)", fontsize=12)
plt.ylabel("Selling Price (EGP)", fontsize=12)
plt.title("Relationship between Storage and Selling Price", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 7. Rating Distribution - Histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Rating'].dropna(), bins=20, color='gold', edgecolor='black')
plt.xlabel("Rating", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Rating Distribution", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#8. Most Expensive Model for Each Brand - Bar Chart
brand_model_price = (
    df.groupby(["Brands", "Models"])['Selling Price'].mean()
    .reset_index(name="average_price")
)

top_price_each_brand = (
    brand_model_price
    .sort_values(["Brands", "average_price"], ascending=[True, False])
    .drop_duplicates(subset="Brands", keep="first")
    .sort_values("average_price", ascending=False)
    .head(10)
)
top_price_each_brand["Brand_Model"] = (top_price_each_brand["Brands"] + 
                                        " - " + top_price_each_brand["Models"])

plt.figure(figsize=(14, 6))
plt.bar(top_price_each_brand["Brand_Model"], top_price_each_brand["average_price"], 
        color='teal', edgecolor='black')
plt.xlabel("Model", fontsize=12)
plt.ylabel("Average Selling Price (EGP)", fontsize=12)
plt.title("Most Expensive Model for Each Brand (Top 10)", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#9. Highest Rated Model for Each Brand - Bar Chart
brand_mode_rating = (
    df.groupby(['Brands', 'Models'])['Rating'].mean()
    .reset_index(name='average_rating')
)

top_rating_each_brand = (
    brand_mode_rating
    .sort_values(['Brands', 'average_rating'], ascending=[True, False])
    .drop_duplicates(subset='Brands', keep='first')
    .sort_values('average_rating', ascending=False)
    .head(10)
)

top_rating_each_brand['Brand_Model'] = (top_rating_each_brand['Brands'] + ' - ' + top_rating_each_brand['Models'])

plt.figure(figsize=(14, 6))
plt.bar(top_rating_each_brand['Brand_Model'], top_rating_each_brand['average_rating'], 
        color='pink', edgecolor='black')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.title('Highest Rated Model for Each Brand (Top 10)', fontsize=14, fontweight='bold')
plt.xticks(rotation=70, ha='right')
plt.tight_layout()
plt.show()




# OUTLIER DETECTION - BOXPLOTS



plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
sns.boxplot(x=df['Selling Price'], color='lightblue')
plt.title('Selling Price - Outlier Detection', fontsize=12, fontweight='bold')
plt.xlabel('Selling Price (EGP)')

plt.subplot(2, 3, 2)
sns.boxplot(x=df['Original Price'], color='lightcoral')
plt.title('Original Price - Outlier Detection', fontsize=12, fontweight='bold')
plt.xlabel('Original Price (EGP)')

plt.subplot(2, 3, 3)
sns.boxplot(x=df['clean_memory'], color='lightgreen')
plt.title('Memory (RAM) - Outlier Detection', fontsize=12, fontweight='bold')
plt.xlabel('Memory (GB)')

plt.subplot(2, 3, 4)
sns.boxplot(x=df['clean_storage'], color='gold')
plt.title('Storage - Outlier Detection', fontsize=12, fontweight='bold')
plt.xlabel('Storage (GB)')

plt.subplot(2, 3, 5)
sns.boxplot(x=df['Rating'], color='pink')
plt.title('Rating - Outlier Detection', fontsize=12, fontweight='bold')
plt.xlabel('Rating')

plt.tight_layout()
plt.show()


# ENCODING - LABEL ENCODING

print("\n Encoding categorical variables...")

encoders = {}

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder
    print(f" Encoded: {col}")

#Handle Rating (Ordinal)
df['Rating'].fillna(df['Rating'].median(), inplace=True)
df['Rating'] = round(df['Rating']) - 1

print("\n Encoding completed!")


#6 STANDARDIZATION

print("\n Applying Standardization (StandardScaler)...")

scaler = StandardScaler()

numeric_features = ['Selling Price', 'Original Price', 'Rating', 'clean_memory', 'clean_storage']

#Apply Standardization
df[numeric_features] = scaler.fit_transform(df[numeric_features])

print("Standardization completed!")
print(f"\nMean of scaled features:\n{df[numeric_features].mean().round(4)}")
print(f"\nStd of scaled features:\n{df[numeric_features].std().round(4)}")


#7 SAVE CLEANED DATA

df.to_csv('Cleaned_Sales_Final.csv', index=False)

print("\n" + "=" * 50)
print(" DATA PREPROCESSING COMPLETED!")
print("=" * 50)
print(f" File saved as: Cleaned_Sales_Final.csv")
print(f" Total rows: {len(df)}")
print(f" Total columns: {len(df.columns)}")
print(f" Columns: {list(df.columns)}")
print("=" * 50)