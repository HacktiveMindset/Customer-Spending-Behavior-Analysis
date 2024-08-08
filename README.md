# Customer Spending Behavior Analysis

![Kaggle Badge](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=fff&style=for-the-badge) ![Jupyter Badge](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=fff&style=for-the-badge) ![pandas Badge](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=fff&style=for-the-badge) ![Microsoft Excel Badge](https://img.shields.io/badge/Microsoft%20Excel-217346?logo=microsoftexcel&logoColor=fff&style=for-the-badge)

For inquiries or feedback, please contact: 

[![INSTAGRAM](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/piyush.mujmule) ![Kaggle Badge](https://img.shields.io/badge/Kaggle-yellow?logo=kaggle&logoColor=fff&style=for-the-badge)

## Project Overview

This project focuses on analyzing customer spending behavior to derive insights and identify patterns. It includes various visualizations and models to enhance understanding of customer spending patterns across demographics, transaction types, times, and locations.

## Objectives

- **Analyze Customer Spending Patterns**: Examine how spending varies across different demographic segments and merchant categories.
- **Identify High-Value Customer Segments**: Determine which customer segments have significant spending and specific preferences.
- **Explore Temporal and Geospatial Patterns**: Investigate spending trends over time and geographical locations.
- **Predict Customer Behavior**: Use machine learning models to predict behavior and detect anomalies.

## Data Description

The dataset contains transaction records with the following columns:

- **`Unnamed: 0`**: Index (not used in analysis).
- **`trans_date_trans_time`**: Transaction date and time.
- **`cc_num`**: Credit card number (anonymized).
- **`merchant`**: Merchant name.
- **`category`**: Merchant category.
- **`amt`**: Transaction amount.
- **`first`**: Customer’s first name.
- **`last`**: Customer’s last name.
- **`gender`**: Customer’s gender.
- **`street`**: Customer’s street address.
- **`city`**: Customer’s city.
- **`state`**: Customer’s state.
- **`zip`**: Customer’s zip code.
- **`lat`**: Latitude of the transaction location.
- **`long`**: Longitude of the transaction location.
- **`city_pop`**: Population of the city.
- **`job`**: Customer’s job title.
- **`dob`**: Customer’s date of birth.
- **`trans_num`**: Unique transaction number.
- **`unix_time`**: Unix timestamp of the transaction.
- **`merch_lat`**: Latitude of the merchant location.
- **`merch_long`**: Longitude of the merchant location.
- **`is_fraud`**: Fraudulent transaction indicator (1 for fraud, 0 for non-fraud).
- **`merch_zipcode`**: Merchant’s zip code.

## Key Analyses and Visualizations

### 1. Customer Spending by Age Group

**Description**: Analyzes total spending across different age groups.

**Code**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/transactions.csv')

# Convert date column to datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# Extract age from date of birth
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = (pd.Timestamp.now() - df['dob']).astype('<m8[Y]')

# Plot
plt.figure(figsize=(10, 6))
df.groupby('age')['amt'].sum().plot(kind='bar')
plt.title('Total Spending by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Total Spending')
plt.show()
```

### 2. Spending Patterns by Gender

**Description**: Compares spending distributions between genders.

**Code**:
```python
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='amt', data=df)
plt.title('Spending Patterns by Gender')
plt.xlabel('Gender')
plt.ylabel('Spending Amount')
plt.show()
```

### 3. Heatmap of Spending by City and State

**Description**: Displays spending intensity across various cities and states.

**Code**:
```python
import seaborn as sns

# Aggregating data
heatmap_data = df.groupby(['state', 'city'])['amt'].sum().unstack()

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.1f')
plt.title('Heatmap of Spending by City and State')
plt.xlabel('City')
plt.ylabel('State')
plt.show()
```

### 4. Spending Trends Over Time

**Description**: Tracks how spending trends change over different months.

**Code**:
```python
df['month'] = df['trans_date_trans_time'].dt.to_period('M')
monthly_spending = df.groupby('month')['amt'].sum()

plt.figure(figsize=(12, 6))
monthly_spending.plot(kind='line')
plt.title('Spending Trends Over Time')
plt.xlabel('Month')
plt.ylabel('Total Spending')
plt.show()
```

### 5. Merchant Category Spending Analysis

**Description**: Breaks down spending by merchant category.

**Code**:
```python
plt.figure(figsize=(10, 6))
df.groupby('category')['amt'].sum().plot(kind='pie', autopct='%1.1f%%')
plt.title('Spending Distribution by Merchant Category')
plt.ylabel('')
plt.show()
```

### 6. Spending Correlation with City Population

**Description**: Analyzes the relationship between spending and city population.

**Code**:
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='city_pop', y='amt', data=df)
plt.title('Spending Correlation with City Population')
plt.xlabel('City Population')
plt.ylabel('Spending Amount')
plt.show()
```

### 7. Geospatial Spending Analysis

**Description**: Visualizes spending across different geographical locations.

**Code**:
```python
import plotly.express as px

fig = px.scatter_geo(df, lat='lat', lon='long', color='amt', size='amt', 
                     hover_name='city', color_continuous_scale='Viridis')
fig.update_layout(title='Geospatial Spending Analysis')
fig.show()
```

### 8. Customer Spending Behavior Over Time

**Description**: Examines changes in customer spending behavior over time.

**Code**:
```python
df['week'] = df['trans_date_trans_time'].dt.to_period('W')
weekly_spending = df.groupby('week')['amt'].sum()

plt.figure(figsize=(12, 6))
weekly_spending.plot(kind='line')
plt.title('Customer Spending Behavior Over Time')
plt.xlabel('Week')
plt.ylabel('Total Spending')
plt.show()
```

### 9. Customer Lifetime Value Analysis

**Description**: Calculates and visualizes the total lifetime value of each customer.

**Code**:
```python
customer_lifetime_value = df.groupby('cc_num')['amt'].sum()

plt.figure(figsize=(12, 6))
customer_lifetime_value.plot(kind='bar')
plt.title('Customer Lifetime Value Analysis')
plt.xlabel('Customer ID')
plt.ylabel('Total Lifetime Value')
plt.show()
```

### 10. Customer Spending by Job Role

**Description**: Analyzes spending patterns by job role.

**Code**:
```python
plt.figure(figsize=(12, 6))
df.groupby('job')['amt'].sum().plot(kind='bar')
plt.title('Spending by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Total Spending')
plt.show()
```

### 11. Spending Comparison Across Different States

**Description**: Compares spending distributions across different states.

**Code**:
```python
plt.figure(figsize=(12, 6))
sns.boxplot(x='state', y='amt', data=df)
plt.title('Spending Comparison Across Different States')
plt.xlabel('State')
plt.ylabel('Spending Amount')
plt.show()
```

### 12. Transaction Amount Distribution

**Description**: Visualizes the distribution of transaction amounts.

**Code**:
```python
plt.figure(figsize=(12, 6))
sns.histplot(df['amt'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.show()
```

### 13. Spending by Zip Code

**Description**: Analyzes total spending by zip code.

**Code**:
```python
plt.figure(figsize=(12, 6))
df.groupby('zip')['amt'].sum().plot(kind='bar')
plt.title('Spending by Zip Code')
plt.xlabel('Zip Code')
plt.ylabel('Total Spending')
plt.show()
```

### 14. Correlation Matrix of Numerical Features

**Description**: Shows correlations between numerical features.

**Code**:
```python
correlation_matrix = df[['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()
```

### 15. Customer Spending Patterns Over Days of the Week

**Description**: Examines how spending varies across different days of the week.

**Code**:
```python
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()
daily_spending = df.groupby('day_of_week')['amt'].sum()

plt.figure(figsize=(12, 6))
daily_spending.plot(kind='bar')
plt.title('Customer Spending Patterns Over Days of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Spending')
plt.show()
```

### 16. Customer Age vs. Transaction Amount Over Time

**Description**: Compares spending patterns by age over time.

**Code**:
```python
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.scatterplot(x='age', y='amt', hue='trans_date_trans_time', data=df, palette='viridis')
plt.title('Customer Age vs. Transaction Amount Over Time')
plt.xlabel('Age

')
plt.ylabel('Transaction Amount')
plt.show()
```

### 17. Geospatial Spending Analysis

**Description**: Interactive map showing spending across different locations.

**Code**:
```python
import plotly.express as px

fig = px.scatter_geo(df, lat='lat', lon='long', color='amt', size='amt', 
                     hover_name='city', title='Geospatial Spending Analysis', 
                     color_continuous_scale='Viridis')
fig.update_layout(title='Geospatial Spending Analysis')
fig.show()
```

### 18. Customer Spending Anomalies Detection

**Description**: Identifies unusual spending patterns using anomaly detection.

**Code**:
```python
from sklearn.ensemble import IsolationForest

# Prepare data for anomaly detection
features = df[['amt']].fillna(0)
model = IsolationForest(contamination=0.01)
df['anomaly'] = model.fit_predict(features)

# Plot anomalies
plt.figure(figsize=(12, 6))
plt.scatter(df.index, df['amt'], c=df['anomaly'], cmap='coolwarm')
plt.title('Customer Spending Anomalies Detection')
plt.xlabel('Transaction Index')
plt.ylabel('Spending Amount')
plt.show()
```

### 19. Spending by Transaction Type

**Description**: Compares spending by different transaction types.

**Code**:
```python
plt.figure(figsize=(12, 6))
df.groupby('category')['amt'].sum().plot(kind='bar', stacked=True)
plt.title('Spending by Transaction Type')
plt.xlabel('Transaction Type')
plt.ylabel('Total Spending')
plt.show()
```

### 20. Interactive Spending Dashboard

**Description**: Provides an interactive dashboard for exploring spending data.

**Code**:
```python
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Create a sample plot
fig = px.line(df, x='trans_date_trans_time', y='amt', title='Spending Over Time')

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Customer Spending Analysis Dashboard'),
    dcc.Graph(
        id='spending-over-time',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
```

## Tools and Technologies

- **Python**: Programming language used for data analysis and visualization.
- **Pandas**: Data manipulation and analysis library.
- **Matplotlib/Seaborn**: Libraries for static data visualization.
- **Plotly**: Library for interactive visualizations.
- **Dash**: Framework for building interactive dashboards.
- **Scikit-Learn**: Library for machine learning and anomaly detection.
- **Lifelines**: Library for survival analysis and churn prediction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
