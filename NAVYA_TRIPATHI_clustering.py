#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


customers = pd.read_csv('C:/Users/tripa/Downloads/Customers.csv')


# In[5]:


products = pd.read_csv('C:/Users/tripa/Downloads/Products.csv')
transactions = pd.read_csv('C:/Users/tripa/Downloads/Transactions.csv')


# In[6]:


print("Customers Dataset:")
print(customers.info(), "\n")
print(customers.head(), "\n")


# In[7]:


print("Products Dataset:")
print(products.info(), "\n")
print(products.head(), "\n")


# In[8]:


print("Transactions Dataset:")
print(transactions.info(), "\n")
print(transactions.head(), "\n")


# In[9]:


# Check for missing values
print("Missing Values:")
print("Customers:", customers.isnull().sum())
print("Products:", products.isnull().sum())
print("Transactions:", transactions.isnull().sum(), "\n")


# In[10]:


# Check for duplicates
print("Duplicates:")
print("Customers:", customers.duplicated().sum())
print("Products:", products.duplicated().sum())
print("Transactions:", transactions.duplicated().sum(), "\n")


# Now we will explore the customers 
# 

# In[11]:


# Customer Signup Trends
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
customers['SignupYear'] = customers['SignupDate'].dt.year
signup_trends = customers['SignupYear'].value_counts().sort_index()


# In[12]:


plt.figure(figsize=(8, 5))
signup_trends.plot(kind='bar', color='skyblue')
plt.title('Customer Signup Trends')
plt.xlabel('Year')
plt.ylabel('Number of Customers')
plt.show()


# In[13]:


# Region-wise distribution of customers
region_dist = customers['Region'].value_counts()
plt.figure(figsize=(8, 5))
region_dist.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Region-wise Distribution of Customers')
plt.ylabel('')
plt.show()


# Now explore the products 
# 

# In[14]:


category_count = products['Category'].value_counts()


# In[15]:


plt.figure(figsize=(8, 5))
category_count.plot(kind='bar', color='coral')
plt.title('Product Category Distribution')
plt.xlabel('Category')
plt.ylabel('Number of Products')
plt.show()


# In[16]:


# Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(products['Price'], bins=20, kde=True, color='teal')
plt.title('Product Price Distribution')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()


# Now explore the transactions 
# 

# In[17]:


transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])


# In[18]:


transactions['Month'] = transactions['TransactionDate'].dt.to_period('M')
monthly_sales = transactions.groupby('Month')['TotalValue'].sum()


# In[19]:


plt.figure(figsize=(10, 6))
monthly_sales.plot(color='purple')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales (USD)')
plt.show()


# In[20]:


# Top products by sales
top_products = transactions.groupby('ProductID')['TotalValue'].sum().sort_values(ascending=False).head(10)
top_products = top_products.reset_index().merge(products, on='ProductID')


# In[21]:


plt.figure(figsize=(10, 6))
sns.barplot(data=top_products, x='ProductName', y='TotalValue', palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Products by Sales')
plt.xlabel('Product Name')
plt.ylabel('Total Sales (USD)')
plt.show()


# In[22]:


high_value_customers = transactions.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False).head(10)
high_value_customers = high_value_customers.reset_index().merge(customers, on='CustomerID')


# In[23]:


print("Top 10 High-Value Customers:")
print(high_value_customers)


# //TASK 2 LOOKALIKE MODEL GENERATION 

# ### Task 2: Lookalike Model
# The goal of the Lookalike Model is to recommend three similar customers for each user based on their profile and transaction history. This is achieved through the following steps:
# 
# 1. Data Preparation
#     I have merged the customer, product and transaction datasets.
#     Then aggregated transaction data to create customer profiles with features such as total spend, quantity purchased, average transaction value and the most purchased product category.
# 
# 2. Feature Normalization:
#   I have then  normalized the numerical features to ensure consistency in scale.
# 
# 3. Similarity Calculation
#    Then computed pairwise customer similarity using the cosine similarity metric.
# 
# 4. Recommendation 
#     Identified the top three most similar customers for each user (excluding themselves).
# 
# 5. Visualization 
#    A heatmap of the similarity matrix was plotted to illustrate the similarity relationships between customers.
# 
# 6. Output:
#     The results are stored in a CSV file with the following structure:
#      CustomerID _:The target customer ID.
#      Lookalike - :  list of tuples containing recommended customer IDs and their similarity scores.
# 

# In[24]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# In[25]:


customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')


# In[26]:


#  Merge datasets
transactions = transactions.merge(products, on='ProductID', how='left')
transactions = transactions.merge(customers, on='CustomerID', how='left')


# In[33]:


products = pd.read_csv('Products.csv')
print(products.columns)  # Check column names in Products.csv
print(products.head())   # Preview the data to ensure the Category column exists


# In[34]:


transactions = transactions.merge(
    products[['ProductID', 'Category', 'Price']],  # Ensure Category is included
    on='ProductID',
    how='left'
)
print(transactions.columns)  # Verify if Category is included
print(transactions.head())   # Preview merged transactions data


# transactions = transactions.merge(products, on='ProductID', how='left')
# transactions = transactions.merge(customers, on='CustomerID', how='left')
# 

# In[101]:


print(products.columns)  # Check column names in Products.csv
print(products.head())   # Preview the data to ensure the Category column exists


# In[102]:


transactions = transactions.merge(
    products[['ProductID', 'Category', 'Price']],  # Ensure Category is included
    on='ProductID',
    how='left'
)


# In[103]:


print(transactions.columns)  # Verify if Category is included
print(transactions.head())   # Preview merged transactions data


# In[104]:


# Drop redundant columns after the merge
transactions_cleaned = transactions.drop(columns=[
    'ProductCategory', 'CustomerPrice', 'Region_x', 'SignupDate_x',
    'ProductName_x', 'Category_x', 'CustomerPrice', 'CustomerName_x',
    'Region_y', 'SignupDate_y', 'ProductName_y', 'Category_y', 'Price_x',
    'CustomerName_y', 'Region_x', 'SignupDate_x', 'ProductName_x', 'Category_x'
])

# Check the cleaned dataframe
print(transactions_cleaned.columns)
print(transactions_cleaned.head())


# In[105]:


# Rename columns for clarity
transactions_cleaned = transactions_cleaned.rename(columns={
    'Category_y': 'Category',  # If this column contains the relevant category
    'Price_y': 'Price'         # If this is the correct price column
})

# Verify the changes
print(transactions_cleaned.columns)
print(transactions_cleaned.head())


# In[108]:


# Step 2: Create Customer Profiles
customer_profiles = transactions_cleaned.groupby('CustomerID').agg({
    'TotalValue': 'sum',    # Total spend
    'Quantity': 'sum',      # Total quantity purchased
    'Price': 'mean',        # Average transaction value
    'Category': lambda x: x.value_counts().index[0] if not x.empty else 'Unknown',  # Most frequent product category
    #'CustomerID': 'first',  # Assuming CustomerName doesn't change
}).reset_index()

# Check the first few rows to ensure the aggregation worked
print(customer_profiles.head())


# In[109]:


print(customer_profiles.head())


# In[110]:


from sklearn.preprocessing import StandardScaler

# Normalize the features (TotalValue, Quantity, Price)
scaler = StandardScaler()
customer_profiles[['TotalValue', 'Quantity', 'Price']] = scaler.fit_transform(customer_profiles[['TotalValue', 'Quantity', 'Price']])

# Check the normalized profiles
print(customer_profiles.head())


# In[111]:


from sklearn.metrics.pairwise import cosine_similarity

# Extract numerical features for similarity calculation
features = customer_profiles[['TotalValue', 'Quantity', 'Price']]

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(features)

# Convert similarity matrix to a DataFrame for easier inspection
similarity_df = pd.DataFrame(similarity_matrix, index=customer_profiles['CustomerID'], columns=customer_profiles['CustomerID'])

# Check the similarity matrix
print(similarity_df.head())


# In[112]:


# Function to get top 3 most similar customers
def get_top_3_similar(customers, similarity_df):
    recommendations = {}
    for customer in customers:
        # Get similarity scores for this customer, excluding self-comparison
        similarity_scores = similarity_df[customer].drop(customer)
        top_3_similar = similarity_scores.nlargest(3)
        recommendations[customer] = list(top_3_similar.items())
    return recommendations

# Get recommendations for all customers
lookalike_recommendations = get_top_3_similar(customer_profiles['CustomerID'], similarity_df)

# Example of lookalike recommendation for the first customer
print(lookalike_recommendations[customer_profiles['CustomerID'].iloc[0]])


# In[113]:


# Prepare the output in the required format
output_data = []
for customer, lookalikes in lookalike_recommendations.items():
    output_data.append({
        'CustomerID': customer,
        'Lookalike': lookalikes
    })

# Convert to DataFrame
output_df = pd.DataFrame(output_data)

# Save the recommendations to CSV
output_df.to_csv('lookalike_recommendations.csv', index=False)

print("Lookalike recommendations saved to 'lookalike_recommendations.csv'")


# In[115]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('lookalike_recommendations.csv')

# Display the first few rows of the DataFrame
df.head()


# In[116]:


# Load the lookalike recommendations CSV file
lookalike_df = pd.read_csv('lookalike_recommendations.csv')

# Display the recommendations for the first 20 customers
print(lookalike_df.head(20))


# In[118]:


true_similar_customers = {
    'C0001': ['C0002', 'C0003', 'C0004'],
    # For other customers, ensure the customer IDs match exactly!
}


# In[119]:


# Check the top 3 lookalikes for customer C0001 and their similarity scores
customer_lookalikes = lookalike_df[lookalike_df['CustomerID'] == 'C0001']['Lookalike'].iloc[0]
print(f"Top 3 Lookalikes for C0001: {customer_lookalikes}")


# In[121]:


# Adjusted code to iterate over the similarity matrix
similarity_threshold = 0.3
lookalike_recommendations = {}

# Get customer IDs (assuming you have a list of customer IDs in the same order as the similarity matrix)
customer_ids = customer_profiles['CustomerID'].tolist()

for i, customer in enumerate(customer_ids):
    # Get similarity scores for the current customer
    similarities = similarity_matrix[i]
    
    # Pair customer IDs with their similarity scores
    similar_customers = [(customer_ids[j], similarities[j]) for j in range(len(similarities)) if i != j and similarities[j] >= similarity_threshold]
    
    # Sort by similarity score and take top 3 recommendations
    lookalike_recommendations[customer] = sorted(similar_customers, key=lambda x: x[1], reverse=True)[:3]

# Now, lookalike_recommendations contains the top 3 most similar customers for each customer


# In[122]:


import pandas as pd

# Prepare the output data for saving
output_data = []
for customer, lookalikes in lookalike_recommendations.items():
    # Format the list of tuples for each customer
    formatted_lookalikes = [(str(cust_id), score) for cust_id, score in lookalikes]
    
    # Add it to the output data
    output_data.append({
        'CustomerID': customer,
        'Lookalike': str(formatted_lookalikes)  # Convert list to string format for CSV
    })

# Convert to DataFrame
output_df = pd.DataFrame(output_data)

# Save the recommendations to CSV
output_df.to_csv('lookalike_recommendations.csv', index=False)

print("Lookalike recommendations saved to 'lookalike_recommendations.csv'")


# ## TASK 3

# In[125]:


import pandas as pd
import numpy as np

# Load the data
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Merge Customers and Transactions data
data = pd.merge(transactions, customers, on='CustomerID', how='left')

# Check for missing values and handle them
data.isnull().sum()

# Convert 'SignupDate' to datetime format
data['SignupDate'] = pd.to_datetime(data['SignupDate'])

# Feature engineering: Aggregate data to create customer profiles
customer_profiles = data.groupby('CustomerID').agg({
    'TotalValue': 'sum',   # Total spend
    'Quantity': 'sum',     # Total quantity purchased
    'Price': 'mean',       # Average transaction value
    'Region': 'first',     # Customer region (assuming it doesn't change)
    'SignupDate': 'first'  # Customer signup date
}).reset_index()

# Calculate the recency of the customer: how long ago they signed up
customer_profiles['Recency'] = (pd.to_datetime('today') - customer_profiles['SignupDate']).dt.days

# Normalize the numerical features (TotalValue, Quantity, Price, Recency)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['TotalValue', 'Quantity', 'Price', 'Recency']
customer_profiles[numerical_features] = scaler.fit_transform(customer_profiles[numerical_features])

# Check the resulting dataset
print(customer_profiles.head())


# In[127]:


from sklearn.cluster import DBSCAN

# Use DBSCAN for clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
customer_profiles['Cluster'] = dbscan.fit_predict(customer_profiles[numerical_features])

# Check the resulting clusters
print(customer_profiles[['CustomerID', 'Cluster']].head())


# In[128]:


# Count the number of clusters (excluding noise points)
num_clusters = len(set(customer_profiles['Cluster']) - {-1})
print(f"Number of clusters formed: {num_clusters}")


# In[129]:


from sklearn.metrics import davies_bouldin_score

# Extract features for DBSCAN clustering (numerical features only)
X = customer_profiles[numerical_features]

# Calculate DB Index
db_index = davies_bouldin_score(X, customer_profiles['Cluster'])
print(f"DB Index: {db_index}")


# In[130]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Perform t-SNE to reduce dimensions to 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=customer_profiles['Cluster'], cmap='viridis', alpha=0.7)
plt.title("Customer Segments (DBSCAN Clustering)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label="Cluster Label")
plt.show()


# In[131]:


from sklearn.decomposition import PCA

# Perform PCA to reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=customer_profiles['Cluster'], cmap='viridis', alpha=0.7)
plt.title("Customer Segments (DBSCAN Clustering)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.show()


# In[132]:


# Analyze cluster sizes
cluster_sizes = customer_profiles['Cluster'].value_counts()
print(f"Cluster Sizes:\n{cluster_sizes}")

# Calculate the mean values for each cluster
cluster_means = customer_profiles.groupby('Cluster')[numerical_features].mean()
print(f"Cluster Mean Profiles:\n{cluster_means}")


# In[133]:


# Save the cluster assignments to a CSV file
customer_profiles[['CustomerID', 'Cluster']].to_csv('customer_clusters.csv', index=False)


# In[ ]:




