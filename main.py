import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snsp

df = pd.read_csv('Sales.csv')

# df.rename(columns={"Selling Price" : "Selling Price (EGP)"}, inplace =True )
# print(df.columns)
# df.rename(columns={"Original Price" : "Original Price (EGP)"}, inplace =True )
# print(df.columns)
# print(df.isnull().sum())


# print(df.isnull().sum())
# df = df.dropna()
# print(df.isnull().sum())

# df = df.drop_duplicates()

df['clean_memory'] = (df['Memory'].str.replace('GB' , '' , case=False).str.replace('MB' , '' , case=False).str.strip())
df['clean_memory'] = pd.to_numeric(df['clean_memory'] , errors='coerce')

df['clean_storage'] = (df['Storage'].str.replace('GB' , '' , case=False).str.replace('MB' , '' , case=False).str.strip())
df['clean_storage'] = pd.to_numeric(df['clean_storage'] , errors='coerce')



df = df.drop(columns=['Memory' , 'Storage'] , axis=1)
print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())
print(df.shape)
df.to_csv('cleaned_sales_data.csv' , index=False)
print(df.describe())

df = pd.read_csv('cleaned_sales_data.csv')

# snsp.boxplot(x=df['Selling Price'])
# plt.show()
# snsp.boxplot(x=df['Original Price'])
# plt.show()
# snsp.boxplot(x=df['clean_memory'])
# plt.show()  
# snsp.boxplot(x=df['clean_storage'])
# plt.show()

#Top 10 most frequent brands 
# brand_counts = df["Brands"].value_counts()

# print(brand_counts.head(10)) 

# plt.figure(figsize=(10, 5))
# brand_counts.head(10).plot(kind="bar")
# plt.xlabel("Brand")
# plt.ylabel("Number of Phones")
# plt.title("Top 10 Most Frequent Brands in Dataset")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

#The most model sold for each brand

# brand_model_counts = (
#     df.groupby(["Brands", "Models"]).size()
#     .reset_index(name="count")
# )
# top_model_each_brand = (
#     brand_model_counts
#     .sort_values(["Brands", "count"], ascending=[True, False])
#     .drop_duplicates(subset="Brands", keep="first")
#     .sort_values("count", ascending=False)   
# )
# top_model_each_brand['Brand_Model'] = top_model_each_brand['Brands'] + ' - ' + top_model_each_brand['Models']
# plt.figure(figsize=(12, 6))
# plt.bar(top_model_each_brand["Brand_Model"], top_model_each_brand["count"])
# plt.xlabel("Model")
# plt.ylabel("Top Model Count")
# plt.title("Best-Selling Model for Each Brand")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

#The most model price for each brand

# brand_model_price = (
#     df.groupby(["Brands", "Models"])['Selling Price'].mean()
#     .reset_index(name="average_price")
# )

# top_price_each_brand = (
#     brand_model_price
#     .sort_values(["Brands", "average_price"], ascending=[True, False])
#     .drop_duplicates(subset="Brands", keep="first")
#     .sort_values("average_price", ascending=False)
# )
# top_price_each_brand["Brand_Model"] = top_price_each_brand["Brands"] + " - " + top_price_each_brand["Models"]

# plt.figure(figsize=(12, 6))
# plt.bar(top_price_each_brand["Brand_Model"], top_price_each_brand["average_price"])
# plt.xlabel("Model") 
# plt.ylabel("Average Selling Price (EGP)")
# plt.title("Most Expensive Model for Each Brand")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# The most model rating for each brand

# brand_mode_rating = (
#     df.groupby(['Brands' , 'Models'])['Rating'].mean()
#     .reset_index(name='average_rating')
# )

# top_rating_each_brand = (
#     brand_mode_rating
#     .sort_values(['Brands' , 'average_rating'] , ascending=[True , False])
#     .drop_duplicates(subset='Brands' , keep='first')
#     .sort_values('average_rating' , ascending=False)
# )

# top_rating_each_brand['Brand_Model'] = top_rating_each_brand['Brands'] + ' - ' + top_rating_each_brand['Models']
# plt.figure(figsize=(12 , 6))
# plt.bar(top_rating_each_brand['Brand_Model'] , top_rating_each_brand['average_rating'])
# plt.xlabel('Model')
# plt.ylabel('Average Rating')
# plt.title('Highest rated model for each brand')
# plt.xticks(rotation=70)
# plt.tight_layout()
# plt.show()

# The relationship between ram and selling price

# df_ram_price = df.dropna(subset=["clean_memory", "Selling Price"])
# plt.figure(figsize=(12,6))
# plt.bar(df_ram_price["clean_memory"], df_ram_price["Selling Price"])
# plt.xlabel("RAM")
# plt.ylabel("Selling Price")
# plt.title("Relation between RAM and Selling Price")
# plt.show()



    







