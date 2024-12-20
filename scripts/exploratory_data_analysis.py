import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ExploratoryDataAnalysis:
    def __init__(self, filepath):
        """Initialize the class with the dataset file path."""
        self.filepath = filepath
        self.dataset = None
        self.cleaned_data = None
        self.data = pd.read_csv(filepath)


    def load_data(self):
        """Load the dataset from a CSV file."""
        try:
            self.dataset = pd.read_csv(self.filepath)
            print("Dataset loaded successfully.")
        except FileNotFoundError:
            print("File not found. Please provide the correct file path.")
    
    def describe_variables(self):
        """Describe all relevant variables and associated data types."""
        if self.dataset is not None:
            print("Dataset Information:\n")
            print(self.dataset.info())
            print("\nSummary Statistics:\n")
            print(self.dataset.describe())
        else:
            print("Dataset not loaded. Please load the dataset first.")

    def handle_missing_values(self):
        """Handle missing values by replacing them with the column mean."""
        if self.dataset is not None:
            self.cleaned_data = self.dataset.fillna(self.dataset.mean(numeric_only=True))
            print("Missing values replaced with column means.")
        else:
            print("Dataset not loaded. Please load the dataset first.")

    def handle_outliers(self, column_name):
        """Treat outliers by capping them within the 1.5*IQR range."""
        if self.cleaned_data is not None:
            Q1 = self.cleaned_data[column_name].quantile(0.25)
            Q3 = self.cleaned_data[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.cleaned_data[column_name] = np.clip(
                self.cleaned_data[column_name], lower_bound, upper_bound
            )
            print(f"Outliers in '{column_name}' handled.")
        else:
            print("Cleaned data not available. Please handle missing values first.")

    def transform_data(self):
        """Segment users into deciles based on total duration and compute total data."""
        if self.cleaned_data is not None:
            self.cleaned_data['Total Duration'] = self.cleaned_data['Dur. (ms)']
            self.cleaned_data['Total Data'] = (
                self.cleaned_data['Total UL (Bytes)'] + self.cleaned_data['Total DL (Bytes)']
            )
            self.cleaned_data['Decile Class'] = pd.qcut(
                self.cleaned_data['Total Duration'], 10, labels=False
            ) + 1
            decile_summary = self.cleaned_data.groupby('Decile Class')['Total Data'].sum()
            print("Decile summary computed successfully:")
            print(decile_summary)
        else:
            print("Cleaned data not available. Please handle missing values first.")

    def visualize_data(self):
        """Visualize the total data distribution by decile class."""
        if self.cleaned_data is not None:
            sns.barplot(
                x=self.cleaned_data['Decile Class'].unique(),
                y=self.cleaned_data.groupby('Decile Class')['Total Data'].sum(),
            )
            plt.xlabel("Decile Class")
            plt.ylabel("Total Data (Bytes)")
            plt.title("Total Data by Decile Class")
            plt.show()
        else:
            print("Cleaned data not available. Please handle missing values first.")

    def calculate_metrics(self):
        """
        Calculate total duration and total data (DL + UL) for each user.
        """
        self.data["Total Duration"] = self.data.groupby("MSISDN/Number")["Dur. (ms)"].transform("sum")
        self.data["Total Data"] = self.data["Total DL (Bytes)"] + self.data["Total UL (Bytes)"]
        print("Metrics calculation completed.")

    def segment_users_by_deciles(self):
        """
        Segment users into decile classes based on total duration and compute total data per decile.
        """
        # Compute decile classes based on Total Duration
        self.data["Duration Decile"] = pd.qcut(self.data["Total Duration"], 10, labels=False) + 1
        
        # Filter for top five decile classes (6-10)
        top_deciles = self.data[self.data["Duration Decile"] >= 6]

        # Compute total data per decile
        self.decile_summary = (
            top_deciles.groupby("Duration Decile")["Total Data"]
            .sum()
            .sort_index(ascending=False)
            .reset_index()
        )
        print("Decile segmentation completed.")
    
    def save_decile_summary(self, output_file):
        """
        Save the decile summary to a CSV file.
        """
        if self.decile_summary is not None:
            self.decile_summary.to_csv(output_file, index=False)
            print(f"Decile summary saved to {output_file}.")
        else:
            print("Decile summary is not available. Please run the segmentation step first.")

    def clean_data(self):
        """
        Perform data cleaning, including handling missing values and ensuring numeric types.
        """
        required_columns = [
            "Dur. (ms)", "Total DL (Bytes)", "Total UL (Bytes)", "MSISDN/Number"
        ]
        self.data = self.data[required_columns]

        # Convert columns to numeric and handle missing values
        for col in required_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
        
        # Fill NaN values with 0
        self.data.fillna(0, inplace=True)
        print("Data cleaning completed.")


    def plot_decile_summary(self):
        """
        Plot the total data per decile class and print the texts used for plotting.
        """
        if self.decile_summary is not None:
            # Print the texts used for plotting
            print("Duration Decile:", self.decile_summary["Duration Decile"].tolist())
            print("Total Data:", self.decile_summary["Total Data"].tolist())
            
            # Plot the data
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=self.decile_summary["Duration Decile"],
                y=self.decile_summary["Total Data"],
                palette="Blues_d"
            )
            plt.title("Total Data Per Top Decile Class")
            plt.xlabel("Decile Class")
            plt.ylabel("Total Data (Bytes)")
            plt.show()
        else:
            print("No decile summary available to plot. Please run the segmentation step first.")
    def analyze_basic_metrics(self):
        """
        Analyze the basic metrics (mean, median, standard deviation, etc.) in the dataset
        and explain their importance for the global objective.
        """
        if self.cleaned_data is not None:
            # Calculate metrics
            mean_values = self.cleaned_data.mean(numeric_only=True)
            median_values = self.cleaned_data.median(numeric_only=True)
            std_dev = self.cleaned_data.std(numeric_only=True)
            
            # Display insights
            print("=== Basic Metrics Analysis ===")
            print("\nMean Values:")
            print(mean_values)
            print("\nMedian Values:")
            print(median_values)
            print("\nStandard Deviation:")
            print(std_dev)
            
            # Provide insights
            print("\n=== Insights ===")

        else:
            print("Cleaned data not available. Please clean the dataset first.")
