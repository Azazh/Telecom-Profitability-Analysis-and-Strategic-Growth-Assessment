import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

class TelecomUserEngagement:
    def __init__(self, data_path):
        """
        Initialize the class with the dataset path.
        """
        self.data = data_path
        self.aggregated_data = None
        self.normalized_data = None
        self.cluster_labels = None
    def aggregate_metrics(self):
        """
        Aggregate metrics per customer (MSISDN).
        """
        self.aggregated_data = self.data.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',  # Count sessions
            'Dur. (ms)': 'sum',  # Sum session durations
            'Total DL (Bytes)': 'sum',  # Sum download traffic
            'Total UL (Bytes)': 'sum',  # Sum upload traffic
        }).reset_index()

        # Calculate total traffic (download + upload)
        self.aggregated_data['Total Traffic (Bytes)'] = (
            self.aggregated_data['Total DL (Bytes)'] + self.aggregated_data['Total UL (Bytes)']
        )

    def get_top_customers(self, metric, top_n=10):
        """
        Get the top customers for a given metric.
        :param metric: Column name of the metric to rank customers by.
        :param top_n: Number of top customers to return.
        :return: Top customers as a DataFrame.
        """
        if self.aggregated_data is None:
            raise ValueError("Metrics have not been aggregated. Call 'aggregate_metrics()' first.")
        return self.aggregated_data.nlargest(top_n, metric)[['MSISDN/Number', metric]]

    def plot_top_customers(self, metric, title, top_n=10):
        """
        Plot the top customers for a given metric.
        :param metric: Column name of the metric to rank customers by.
        :param title: Title of the plot.
        :param top_n: Number of top customers to display.
        """
        top_customers = self.get_top_customers(metric, top_n)
        plt.figure(figsize=(10, 6))
        plt.bar(top_customers['MSISDN/Number'].astype(str), top_customers[metric], color='skyblue')
        plt.title(title, fontsize=16)
        plt.xlabel("MSISDN (Customer ID)", fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.xticks(rotation=45, fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    def normalize_metrics(self):
        """
        Normalize the metrics using Min-Max scaling.
        """
        if self.aggregated_data is None:
            raise ValueError("Metrics have not been aggregated. Call 'aggregate_metrics()' first.")
        
        scaler = MinMaxScaler()
        metrics = ['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']
        self.normalized_data = self.aggregated_data.copy()
        self.normalized_data[metrics] = scaler.fit_transform(self.aggregated_data[metrics])

    def run_kmeans(self, n_clusters=3):
        """
        Run K-means clustering on the normalized data.
        :param n_clusters: Number of clusters (default is 3).
        """
        if self.normalized_data is None:
            raise ValueError("Metrics have not been normalized. Call 'normalize_metrics()' first.")

        # Select normalized metrics for clustering
        metrics = ['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.normalized_data[metrics])
        self.normalized_data['Cluster'] = self.cluster_labels
    def plot_clusters(self):
        """
        Visualize the clustering results using PCA for dimensionality reduction.
        """
        if self.cluster_labels is None:
            raise ValueError("K-means has not been run. Call 'run_kmeans()' first.")

        # Reduce dimensions using PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.normalized_data[['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']])
        reduced_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
        reduced_df['Cluster'] = self.normalized_data['Cluster']

        # Plot the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='PCA1', y='PCA2', hue='Cluster', data=reduced_df, palette='viridis', s=100
        )
        plt.title("Customer Engagement Clusters", fontsize=16)
        plt.xlabel("Principal Component 1", fontsize=14)
        plt.ylabel("Principal Component 2", fontsize=14)
        plt.legend(title="Cluster", fontsize=12)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()

    def compute_cluster_statistics(self):
            """
            Compute the minimum, maximum, average, and total metrics for each cluster.
            """
            if self.cluster_labels is None:
                raise ValueError("K-means has not been run. Call 'run_kmeans()' first.")

            # Join cluster labels back to the non-normalized data
            merged_data = self.normalized_data[['MSISDN/Number', 'Cluster']].merge(
                self.aggregated_data,
                on='MSISDN/Number'
            )

            # Group by cluster and compute statistics
            self.cluster_statistics = merged_data.groupby('Cluster').agg({
                'Bearer Id': ['min', 'max', 'mean', 'sum'],
                'Dur. (ms)': ['min', 'max', 'mean', 'sum'],
                'Total Traffic (Bytes)': ['min', 'max', 'mean', 'sum']
            })

            # Flatten MultiIndex columns for better readability
            self.cluster_statistics.columns = [
                f"{metric}_{stat}" for metric, stat in self.cluster_statistics.columns
            ]
            self.cluster_statistics.reset_index(inplace=True)

    def aggregate_traffic_per_application(self):
        """
        Aggregate user total traffic per application.
        """
        app_columns = [
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)'
        ]
        
        self.app_traffic_data = self.data.groupby('MSISDN/Number')[app_columns].sum().reset_index()
        self.app_traffic_data['Total Social Media Traffic (Bytes)'] = (
            self.app_traffic_data['Social Media DL (Bytes)'] +
            self.app_traffic_data['Social Media UL (Bytes)']
        )
        self.app_traffic_data['Total Google Traffic (Bytes)'] = (
            self.app_traffic_data['Google DL (Bytes)'] +
            self.app_traffic_data['Google UL (Bytes)']
        )
        self.app_traffic_data['Total Youtube Traffic (Bytes)'] = (
            self.app_traffic_data['Youtube DL (Bytes)'] +
            self.app_traffic_data['Youtube UL (Bytes)']
        )

        # Derive top 10 most engaged users for each application
        self.top_app_users = {
            'Social Media': self.app_traffic_data.nlargest(10, 'Total Social Media Traffic (Bytes)'),
            'Google': self.app_traffic_data.nlargest(10, 'Total Google Traffic (Bytes)'),
            'Youtube': self.app_traffic_data.nlargest(10, 'Total Youtube Traffic (Bytes)')
        }

    def plot_top_applications(self):
        """
        Plot the top 3 most used applications.
        """
        if self.app_traffic_data is None:
            raise ValueError("Application traffic has not been aggregated. Call 'aggregate_traffic_per_application()' first.")
        
        app_totals = {
            'Social Media': self.app_traffic_data['Total Social Media Traffic (Bytes)'].sum(),
            'Google': self.app_traffic_data['Total Google Traffic (Bytes)'].sum(),
            'Youtube': self.app_traffic_data['Total Youtube Traffic (Bytes)'].sum()
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(app_totals.keys(), app_totals.values(), color=['blue', 'green', 'red'])
        plt.title("Top 3 Most Used Applications by Total Traffic")
        plt.ylabel("Total Traffic (Bytes)")
        plt.show()


    def elbow_method(self):
        """
        Find the optimized value of k using the elbow method.
        """
        if self.normalized_data is None:
            raise ValueError("Metrics have not been normalized. Call 'normalize_metrics()' first.")
        
        metrics = ['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']
        sse = []
        k_range = range(1, 11)  # Test k values from 1 to 10
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.normalized_data[metrics])
            sse.append(kmeans.inertia_)
        
        self.elbow_curve_data = {'k': list(k_range), 'sse': sse}

        # Plot the elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, sse, marker='o')
        plt.title("Elbow Method for Optimal k")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Sum of Squared Errors (SSE)")
        plt.show()