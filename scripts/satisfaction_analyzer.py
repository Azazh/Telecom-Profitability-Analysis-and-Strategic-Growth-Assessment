# Import required libraries
import pandas as pd
import numpy as np
import psycopg2
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from IPython.display import display

class SatisfactionAnalyzer:
    def __init__(self, data):
        """
        Initialize the analyzer with the dataset.
        """
        self.dataset = data
        self.engagement_scores = None
        self.experience_scores = None
        self.satisfaction_scores = None
        self.clusters = None

    def preprocess_data(self):
        """
        Preprocess the data by handling missing values and standardizing columns as needed.
        """
        # Fill missing values and normalize data if required
        self.dataset.fillna(0, inplace=True)
        print("Data preprocessed successfully.")

    def calculate_engagement_experience_scores(self):
        """
        Calculate engagement and experience scores using Euclidean distance.
        """
        # Assuming engagement and experience clusters are derived from k-means clustering
        engagement_features = ['Activity Duration DL (ms)', 'Activity Duration UL (ms)']
        experience_features = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)']
        
        kmeans_engagement = KMeans(n_clusters=2, random_state=42)
        self.dataset['EngagementCluster'] = kmeans_engagement.fit_predict(self.dataset[engagement_features])
        engagement_centers = kmeans_engagement.cluster_centers_
        less_engaged_center = engagement_centers[0]  # Assuming cluster 0 is less engaged
        
        kmeans_experience = KMeans(n_clusters=2, random_state=42)
        self.dataset['ExperienceCluster'] = kmeans_experience.fit_predict(self.dataset[experience_features])
        experience_centers = kmeans_experience.cluster_centers_
        worst_experience_center = experience_centers[0]  # Assuming cluster 0 is worst experience
        
        self.dataset['EngagementScore'] = self.dataset[engagement_features].apply(
            lambda x: np.linalg.norm(x - less_engaged_center), axis=1
        )
        self.dataset['ExperienceScore'] = self.dataset[experience_features].apply(
            lambda x: np.linalg.norm(x - worst_experience_center), axis=1
        )

        print("Engagement and experience scores calculated successfully.")

    def calculate_satisfaction_scores(self):
        """
        Calculate the satisfaction score as the average of engagement and experience scores.
        """
        self.dataset['SatisfactionScore'] = (
            self.dataset['EngagementScore'] + self.dataset['ExperienceScore']
        ) / 2
        print("Satisfaction scores calculated successfully.")

    def regression_model(self):
        """
        Build a regression model to predict the satisfaction score.
        """
        features = ['EngagementScore', 'ExperienceScore']
        target = 'SatisfactionScore'
        
        X = self.dataset[features]
        y = self.dataset[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Regression model built successfully with MSE: {mse:.2f}")

    def kmeans_clustering(self):
        """
        Perform k-means clustering on engagement and experience scores.
        """
        kmeans = KMeans(n_clusters=2, random_state=42)
        self.dataset['Cluster'] = kmeans.fit_predict(self.dataset[['EngagementScore', 'ExperienceScore']])
        print("K-means clustering completed successfully.")

    def aggregate_scores(self):
        """
        Aggregate average satisfaction and experience scores per cluster
        and plot the results.
        """
        # Aggregate average scores per cluster
        aggregated = self.dataset.groupby('Cluster').agg({
            'EngagementScore': 'mean',
            'ExperienceScore': 'mean',
            'SatisfactionScore': 'mean'
        })

        print("Aggregated scores per cluster:")
        print(aggregated)

        # Plot the aggregated scores
        self.plot_aggregated_scores(aggregated)

    def plot_aggregated_scores(self, aggregated):
        """
        Plot aggregated scores per cluster.
        """
        aggregated.plot(kind='bar', figsize=(10, 6), colormap='viridis', edgecolor='black')

        # Customize plot
        plt.title("Average Scores per Cluster", fontsize=16)
        plt.xlabel("Cluster", fontsize=14)
        plt.ylabel("Average Score", fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.legend(title="Metrics", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def export_to_postgresql(self, host, database, user, password, table_name):
        """
        Export the dataset with scores to a PostgreSQL database.
        """
        try:
            # Connect to PostgreSQL database
            connection = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            cursor = connection.cursor()

            # Create table if it does not exist
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                UserId NUMERIC,
                EngagementScore FLOAT,
                ExperienceScore FLOAT,
                SatisfactionScore FLOAT,
                Cluster INT
            );
            """)

            # Insert data into the table
            for _, row in self.dataset.iterrows():
                cursor.execute(f"""
                INSERT INTO {table_name} (UserId, EngagementScore, ExperienceScore, SatisfactionScore, Cluster)
                VALUES (%s, %s, %s, %s, %s);
                """, (row['Bearer Id'], row['EngagementScore'], row['ExperienceScore'], row['SatisfactionScore'], row['Cluster']))

            # Commit the transaction
            connection.commit()
            display(f"Data exported to PostgreSQL table '{table_name}' successfully.")

        except psycopg2.Error as e:
            print(f"An error occurred: {e}")
        finally:
            # Close the connection
            if connection:
                cursor.close()
                connection.close()

    def run_analysis(self):
        """
        Execute all tasks sequentially.
        """
        self.preprocess_data()
        self.calculate_engagement_experience_scores()
        self.calculate_satisfaction_scores()
        self.regression_model()
        self.kmeans_clustering()
        self.aggregate_scores()
