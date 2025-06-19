# ---------------------------------------
# E-Commerce Product Performance Analysis
# ---------------------------------------
# Author: Arjun P G
# Description: Analyzing sales, reviews, and delivery patterns from the Olist e-commerce dataset using Python.

# STEP 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot styles for consistency
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# STEP 2: Load the Datasets
# These CSVs should be uploaded to your Colab environment or local working directory
orders = pd.read_csv("olist_orders_dataset.csv")
items = pd.read_csv("olist_order_items_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
reviews = pd.read_csv("olist_order_reviews_dataset.csv")

# STEP 3: Merge Datasets to Create a Unified View
# Join items with product info
order_items = pd.merge(items, products, on='product_id', how='left')
# Merge with orders table
full_data = pd.merge(order_items, orders, on='order_id', how='left')
# Add review scores
full_data = pd.merge(full_data, reviews[['order_id', 'review_score']], on='order_id', how='left')

# STEP 4: Analyze Product-Level Performance
# Group by product and calculate total sales, number of orders, and average review
product_sales = full_data.groupby('product_id').agg({
    'price': 'sum',
    'order_id': 'count',
    'review_score': 'mean'
}).rename(columns={
    'price': 'total_sales',
    'order_id': 'order_count'
}).sort_values(by='total_sales', ascending=False)

print("🔝 Top 5 Best-Selling Products:")
print(product_sales.head())

# STEP 5: Visualize Top Products
top_products = product_sales.head(10)
top_products['total_sales'].plot(kind='bar', title='Top 10 Best-Selling Products')
plt.ylabel('Total Sales (BRL)')
plt.show()

# STEP 6: Analyze Category-Level Performance
# Aggregate sales and reviews by product category
category_performance = full_data.groupby('product_category_name').agg({
    'price': 'sum',
    'order_id': 'count',
    'review_score': 'mean'
}).rename(columns={
    'price': 'category_sales',
    'order_id': 'category_orders'
}).sort_values(by='category_sales', ascending=False)

print("\n🏆 Top 5 Categories by Sales:")
print(category_performance.head())

# STEP 7: Visualize Customer Review Score Distribution
sns.countplot(data=reviews, x='review_score', palette='coolwarm')
plt.title("Customer Review Score Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
plt.show()

# STEP 8: Shipping Delay Analysis
# Convert timestamp columns to datetime
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])

# Calculate delivery delay in days
orders['shipping_delay'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days

# Remove orders where delivery dates are missing (NaT)
orders = orders[orders['shipping_delay'].notnull()]

# Plot distribution of delivery delays
sns.histplot(orders['shipping_delay'], bins=30, kde=True, color='green')
plt.title("Shipping Delay Distribution (in days)")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Orders")
plt.show()

print("\n📦 Shipping Delay Summary Stats:")
print(orders['shipping_delay'].describe())
