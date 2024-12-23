import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

class UserExperienceAnalytics:
    
    def __init__(self, data):
        self.data = data
    
    def clean_data(self):
        # Replace missing values with the mean or mode of the corresponding variable
        self.data['TCP DL Retrans. Vol (Bytes)'].fillna(self.data['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
        self.data['Avg RTT DL (ms)'].fillna(self.data['Avg RTT DL (ms)'].mean(), inplace=True)
        self.data['Avg Bearer TP DL (kbps)'].fillna(self.data['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
        self.data['Handset Type'].fillna(self.data['Handset Type'].mode()[0], inplace=True)
    
    def aggregate_per_customer(self):
        # Convert columns to numeric where appropriate
        numeric_columns = ['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']
        
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Handle missing values (fill NaN with mean for numeric columns)
        for col in numeric_columns:
            self.data[col].fillna(self.data[col].mean(), inplace=True)

        # Aggregate per customer
        aggregated_data = self.data.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Handset Type': 'first',  # Can use 'first' or 'mode' for categorical columns
            'Avg Bearer TP DL (kbps)': 'mean'
        }).reset_index()
        
        return aggregated_data
    
    def top_bottom_most_frequent(self, column, n=10):
        # Ensure column exists
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        
        # Top n values
        top_n = self.data.nlargest(n, column)[column]
        
        # Bottom n values
        bottom_n = self.data.nsmallest(n, column)[column]
        
        # Most frequent n values
        most_frequent_n = self.data[column].value_counts().head(n)
        
        return top_n, bottom_n, most_frequent_n
    
    def average_throughput_per_handset(self):
        # Ensure 'Handset Type' is valid
        if 'Handset Type' not in self.data.columns:
            raise ValueError("Column 'Handset Type' does not exist in the dataset.")
        
        avg_throughput = self.data.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean()
        avg_throughput.plot(kind='bar', figsize=(12, 6))
        plt.title('Average Throughput per Handset Type')
        plt.xlabel('Handset Type')
        plt.ylabel('Average Throughput (kbps)')
        plt.show()
        return avg_throughput

    def average_tcp_retransmission_per_handset(self):
        avg_tcp_retransmission = self.data.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean()
        avg_tcp_retransmission.plot(kind='bar', figsize=(12, 6), color='orange')
        plt.title('Average TCP Retransmission per Handset Type')
        plt.xlabel('Handset Type')
        plt.ylabel('Average TCP Retransmission (Bytes)')
        plt.show()
        return avg_tcp_retransmission

    def kmeans_clustering(self, n_clusters=3):
        experience_metrics = self.data[['Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (Bytes)']]
        
        # Normalize the features before clustering
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        
        scaler = StandardScaler()
        scaled_metrics = scaler.fit_transform(experience_metrics)
        
        kmeans = KMeans(n_clusters=n_clusters)
        self.data['Cluster'] = kmeans.fit_predict(scaled_metrics)
        
        # Plotting the clusters
        plt.figure(figsize=(12, 6))
        plt.scatter(self.data['Avg Bearer TP DL (kbps)'], self.data['TCP DL Retrans. Vol (Bytes)'], c=self.data['Cluster'], cmap='viridis')
        plt.title('K-means Clustering of Users based on Experience Metrics')
        plt.xlabel('Average Throughput (kbps)')
        plt.ylabel('TCP Retransmission (Bytes)')
        plt.show()

        # Describing each cluster
        cluster_description = self.data.groupby('Cluster').mean()
        return cluster_description
