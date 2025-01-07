# 导入必要的库
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

from matplotlib.font_manager import FontProperties, fontManager

# 动态注册字体
font_path = r"C:\Windows\Fonts\SimHei.ttf"
fontManager.addfont(font_path)

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 设置图形的风格
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 读取数据
data = pd.read_csv('Walmart.csv')

# 数据预处理
data_clean = data.dropna()

# 转换日期字段为datetime类型
data_clean['transaction_date'] = pd.to_datetime(data_clean['transaction_date'])

# 添加辅助列，例如年、月、周、日、星期几
data_clean['year'] = data_clean['transaction_date'].dt.year
data_clean['month'] = data_clean['transaction_date'].dt.month
data_clean['week'] = data_clean['transaction_date'].dt.isocalendar().week
data_clean['day'] = data_clean['transaction_date'].dt.day
data_clean['weekday'] = data_clean['transaction_date'].dt.weekday  # 0: Monday, 6: Sunday

# 计算销售额
data_clean['sales'] = data_clean['quantity_sold'] * data_clean['unit_price']

# 可视化部分

# 可视化部分
# 1. Top 10 products by sales
plt.figure(figsize=(12,6))
top_products = data_clean.groupby('product_name')['sales'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.xlabel('Total Sales')
plt.ylabel('Product Name')
plt.title('Top 8 Products by Sales')
plt.show()

# 2. Total sales by category
plt.figure(figsize=(12,6))
category_sales = data_clean.groupby('category')['sales'].sum().sort_values(ascending=False)
sns.barplot(x=category_sales.index, y=category_sales.values, palette='magma')
plt.xticks(rotation=45)
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.title('Total Sales by Category')
plt.show()

# 3. Sales quantity distribution
plt.figure(figsize=(10,6))
sns.histplot(data_clean['quantity_sold'], bins=30, kde=True, color='skyblue')
plt.xlabel('Quantity Sold')
plt.title('Sales Quantity Distribution')
plt.show()

# 4. Unit price distribution
plt.figure(figsize=(10,6))
sns.histplot(data_clean['unit_price'], bins=30, kde=True, color='salmon')
plt.xlabel('Unit Price')
plt.title('Unit Price Distribution')
plt.show()

# 5. Daily sales trend
sales_over_time = data_clean.groupby('transaction_date')['sales'].sum().reset_index()

plt.figure(figsize=(14,7))
plt.plot(sales_over_time['transaction_date'], sales_over_time['sales'], label='Daily Sales', color='green')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Daily Sales Trend')
plt.legend()
plt.show()

# 6. Top 10 stores by sales
plt.figure(figsize=(12,6))
store_sales = data_clean.groupby('store_id')['sales'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=store_sales.values, y=store_sales.index, palette='coolwarm')
plt.xlabel('Total Sales')
plt.ylabel('Store ID')
plt.title('Top 10 Stores by Sales')
plt.show()

# 7. Sales distribution by store location
plt.figure(figsize=(12,6))
location_sales = data_clean.groupby('store_location')['sales'].sum().sort_values(ascending=False)
sns.barplot(x=location_sales.index, y=location_sales.values, palette='inferno')
plt.xticks(rotation=45)
plt.xlabel('Store Location')
plt.ylabel('Total Sales')
plt.title('Sales Distribution by Store Location')
plt.show()

# 8. Inventory level distribution
plt.figure(figsize=(10,6))
sns.histplot(data_clean['inventory_level'], bins=30, kde=True, color='purple')
plt.xlabel('Inventory Level')
plt.title('Inventory Level Distribution')
plt.show()

# 9. Reorder point vs inventory level
plt.figure(figsize=(10,6))
sns.scatterplot(x='inventory_level', y='reorder_point', hue='stockout_indicator', data=data_clean, palette='deep')
plt.xlabel('Inventory Level')
plt.ylabel('Reorder Point')
plt.title('Inventory Level vs Reorder Point')
plt.show()

# 10. Reorder quantity distribution
plt.figure(figsize=(10,6))
sns.histplot(data_clean['reorder_quantity'], bins=30, kde=True, color='orange')
plt.xlabel('Reorder Quantity')
plt.title('Reorder Quantity Distribution')
plt.show()

# 11. Top 10 suppliers by order quantity
plt.figure(figsize=(12,6))
top_suppliers = data_clean.groupby('supplier_id')['transaction_id'].count().sort_values(ascending=False).head(10)
sns.barplot(x=top_suppliers.values, y=top_suppliers.index, palette='cividis')
plt.xlabel('Order Quantity')
plt.ylabel('Supplier ID')
plt.title('Top 10 Suppliers by Order Quantity')
plt.show()

# 12. Supplier lead time distribution
plt.figure(figsize=(10,6))
sns.boxplot(x='supplier_lead_time', data=data_clean, color='teal')
plt.xlabel('Supplier Lead Time (days)')
plt.title('Supplier Lead Time Distribution')
plt.show()

# 13. Customer age distribution
plt.figure(figsize=(10,6))
sns.histplot(data_clean['customer_age'], bins=30, kde=True, color='brown')
plt.xlabel('Customer Age')
plt.title('Customer Age Distribution')
plt.show()

# 14. Customer gender distribution
plt.figure(figsize=(8,6))
gender_counts = data_clean['customer_gender'].value_counts()
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='pastel')
plt.xlabel('Customer Gender')
plt.ylabel('Count')
plt.title('Customer Gender Distribution')
plt.show()

# 15. Customer income distribution
plt.figure(figsize=(10,6))
sns.histplot(data_clean['customer_income'], bins=30, kde=True, color='darkblue')
plt.xlabel('Customer Income')
plt.title('Customer Income Distribution')
plt.show()

# 16. Customer income vs sales
plt.figure(figsize=(10,6))
sns.scatterplot(x='customer_income', y='sales', hue='customer_gender', data=data_clean, alpha=0.6)
plt.xlabel('Customer Income')
plt.ylabel('Sales')
plt.title('Customer Income vs Sales')
plt.show()

# 17. Customer loyalty level
plt.figure(figsize=(10,6))
loyalty_sales = data_clean.groupby('customer_loyalty_level')['sales'].sum().sort_values(ascending=False)
sns.barplot(x=loyalty_sales.index, y=loyalty_sales.values, palette='viridis')
plt.xlabel('Customer Loyalty Level')
plt.ylabel('Total Sales')
plt.title('Total Sales by Customer Loyalty Level')
plt.show()

# 18. Payment method distribution
plt.figure(figsize=(8,6))
payment_counts = data_clean['payment_method'].value_counts()
sns.barplot(x=payment_counts.index, y=payment_counts.values, palette='coolwarm')
plt.xlabel('Payment Method')
plt.ylabel('Transaction Count')
plt.title('Transaction Count by Payment Method')
plt.show()

# 19. Promotion application count
plt.figure(figsize=(8,6))
promo_counts = data_clean['promotion_applied'].value_counts()
sns.barplot(x=promo_counts.index, y=promo_counts.values, palette='Set2')
plt.xlabel('Promotion Applied')
plt.ylabel('Transaction Count')
plt.title('Transaction Count with Promotion Application')
plt.show()

# 20. Promotion type vs sales
plt.figure(figsize=(12,6))
sns.boxplot(x='promotion_type', y='sales', data=data_clean, palette='Set3')
plt.xlabel('Promotion Type')
plt.ylabel('Sales')
plt.title('Sales by Promotion Type')
plt.show()

# 21. Weather conditions vs sales
plt.figure(figsize=(12,6))
weather_sales = data_clean.groupby('weather_conditions')['sales'].sum().sort_values(ascending=False)
sns.barplot(x=weather_sales.index, y=weather_sales.values, palette='Spectral')
plt.xticks(rotation=45)
plt.xlabel('Weather Conditions')
plt.ylabel('Total Sales')
plt.title('Total Sales under Different Weather Conditions')
plt.show()

# 22. Holiday vs non-holiday sales
holiday_sales = data_clean.groupby('holiday_indicator')['sales'].sum()

# Convert boolean index to integer index
holiday_sales.index = holiday_sales.index.map({False: 0, True: 1})

# Debug print holiday sales content
print(holiday_sales)

# Bar plot
plt.figure(figsize=(8,6))
sns.barplot(x=holiday_sales.index.map({0: 'Non-Holiday', 1: 'Holiday'}), y=holiday_sales.values, palette='Accent')
plt.xlabel('Holiday')
plt.ylabel('Total Sales')
plt.title('Comparison of Total Sales on Holidays and Non-Holidays')
plt.show()

# 23. Sales by weekday
plt.figure(figsize=(10,6))
weekday_sales = data_clean.groupby('weekday')['sales'].sum().reset_index()
weekday_sales['weekday_name'] = weekday_sales['weekday'].apply(lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
sns.barplot(x='weekday_name', y='sales', data=weekday_sales, palette='Blues_d')
plt.xlabel('Weekday')
plt.ylabel('Total Sales')
plt.title('Total Sales by Weekday')
plt.show()

# 24. Out-of-stock vs in-stock transactions
stockout_counts = data_clean['stockout_indicator'].value_counts()

# Convert boolean index to integer index
stockout_counts.index = stockout_counts.index.map({False: 'In Stock', True: 'Out of Stock'})

# Debug print stockout counts content
print(stockout_counts)

# Bar plot
plt.figure(figsize=(8,6))
sns.barplot(x=stockout_counts.index, y=stockout_counts.values, palette='Reds')
plt.xlabel('Stock Status')
plt.ylabel('Transaction Count')
plt.title('Transaction Count by Stock Status')
plt.show()

# 25. Correlation heatmap of numeric features
# 25. Correlation heatmap of numeric features
plt.figure(figsize=(16,12))

# 从数据集中选择数值型特征，排除 'year' 列
numeric_features = data_clean.select_dtypes(include=[np.number])
numeric_features = numeric_features.drop(columns=['year'])  # 假设 'year' 是列名
# 计算相关性矩阵
corr = numeric_features.corr()

# 绘制热图
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')

# 修正标题中的拼写错误
plt.title('Correlation Heatmap of Numeric Features')

# 显示图表
plt.show()
