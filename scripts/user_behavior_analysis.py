import pandas as pd
import matplotlib.pyplot as plt

class UserBehaviorAnalysis:
    def __init__(self, dataset_path):
        """Initialize the analysis class with the dataset."""
        self.data = pd.read_csv(dataset_path)
        self.processed_data = None

    def aggregate_user_data(self):
        """Aggregate per-user data for the required columns."""
        self.processed_data = self.data.groupby("MSISDN/Number").agg(
            xdr_sessions=("Dur. (ms)", "count"),
            session_duration=("Dur. (ms)", "sum"),
            total_download=("Total DL (Bytes)", "sum"),
            total_upload=("Total UL (Bytes)", "sum"),
            social_media_data=("Social Media DL (Bytes)", "sum"),
            google_data=("Google DL (Bytes)", "sum"),
            email_data=("Email DL (Bytes)", "sum"),
            youtube_data=("Youtube DL (Bytes)", "sum"),
            netflix_data=("Netflix DL (Bytes)", "sum"),
            gaming_data=("Gaming DL (Bytes)", "sum"),
            other_data=("Other DL (Bytes)", "sum")
        ).reset_index()
        self.processed_data["total_volume"] = (
            self.processed_data["total_download"] + self.processed_data["total_upload"]
        )
        return self.processed_data

    def plot_aggregated_data(self):
        """Plot and display the aggregated data."""
        if self.processed_data is None:
            print("Data is not aggregated yet. Call aggregate_user_data() first.")
            return
        
        # Top 10 users by session duration
        top_users_by_duration = self.processed_data.nlargest(10, "session_duration")
        plt.figure(figsize=(12, 6))
        plt.bar(top_users_by_duration["MSISDN/Number"], top_users_by_duration["session_duration"])
        plt.xlabel("User Number")
        plt.ylabel("Total Session Duration (ms)")
        plt.title("Top 10 Users by Session Duration")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Total download vs upload data
        plt.figure(figsize=(12, 6))
        plt.scatter(
            self.processed_data["total_download"], 
            self.processed_data["total_upload"], 
            alpha=0.7
        )
        plt.xlabel("Total Download (Bytes)")
        plt.ylabel("Total Upload (Bytes)")
        plt.title("Download vs Upload Data per User")
        plt.tight_layout()
        plt.show()
        
        # Total volume per application
        app_data = self.processed_data[
            ["social_media_data", "google_data", "email_data", "youtube_data", 
             "netflix_data", "gaming_data", "other_data"]
        ].sum().sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(app_data.index, app_data.values)
        plt.xlabel("Application")
        plt.ylabel("Total Data Volume (Bytes)")
        plt.title("Total Data Volume per Application")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def print_insights(self):
        """Print key insights based on the analysis."""
        if self.processed_data is None:
            print("Data is not aggregated yet. Call aggregate_user_data() first.")
            return

        # Total sessions and data volume
        total_sessions = self.processed_data["xdr_sessions"].sum()
        total_volume = self.processed_data["total_volume"].sum()
        print(f"Total xDR Sessions: {total_sessions}")
        print(f"Total Data Volume: {total_volume / 1e9:.2f} GB")

        # Top applications by data volume
        app_data = self.processed_data[
            ["social_media_data", "google_data", "email_data", "youtube_data", 
             "netflix_data", "gaming_data", "other_data"]
        ].sum().sort_values(ascending=False)
        print("\nTop Applications by Data Volume:")
        for app, volume in app_data.items():
            print(f"{app}: {volume / 1e9:.2f} GB")

        # Top 5 users by session duration
        top_users = self.processed_data.nlargest(5, "session_duration")
        print("\nTop 5 Users by Session Duration:")
        print(top_users[["MSISDN/Number", "session_duration"]])

# Notebook Instructions
# 1. Save the notebook as "user_behavior_overview.ipynb".
# 2. Load the dataset using the `UserBehaviorAnalysis` class.
# 3. Call the appropriate methods for analysis, plotting, and insights.
