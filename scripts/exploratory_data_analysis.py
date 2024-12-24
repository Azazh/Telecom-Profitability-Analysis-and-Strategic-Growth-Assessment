import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython.display import display

class ExploratoryDataAnalysis:
    def __init__(self, filepath):
        """Initialize the class with the dataset file path."""
        self.filepath = filepath
        self.dataset = None
        self.cleaned_data = None
        self.data = filepath
        self.data.columns = self.data.columns.str.strip()


    def load_data(self):
        """Load the dataset from a CSV file."""
        try:
            self.dataset = self.filepath
            print("Dataset loaded successfully.")
        except FileNotFoundError:
            print("File not found. Please provide the correct file path.")
    
    def describe_variables(self):
        """Describe all relevant variables and associated data types."""
        if self.dataset is not None:
            print("Dataset Information:\n")
            display(self.dataset.info())
            print("\nSummary Statistics:\n")
            display(self.dataset.describe())
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
            display(f"Outliers in '{column_name}' handled.")
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
            display(decile_summary)
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
            display("=== Basic Metrics Analysis ===")
            print("\nMean Values:")
            display(mean_values)
            print("\nMedian Values:")
            display(median_values)
            print("\nStandard Deviation:")
            display(std_dev)
            
            # Provide insights
            print("\n=== Insights ===")

        else:
            print("Cleaned data not available. Please clean the dataset first.")


    def compute_dispersion(self):
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        stats = numeric_data.describe().T
        stats['range'] = stats['max'] - stats['min']
        stats['IQR'] = stats['75%'] - stats['25%']
        return stats

    def plot_univariate(self):
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        for col in numeric_data.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[col], kde=True)
            plt.title(f'Univariate Analysis - {col}')
            plt.show()
    # def graphical_univariate_analysis(self):
        """
        Conduct graphical univariate analysis by plotting histograms and boxplots
        for each quantitative variable.
        """
        if self.cleaned_data is not None:
            for column in self.cleaned_data.select_dtypes(include=['float64', 'int64']).columns:
                # Plot histogram
                plt.figure(figsize=(10, 6))
                sns.histplot(self.cleaned_data[column], kde=True, bins=20, color="blue")
                plt.title(f"Histogram of {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.show()
                
                # Plot boxplot
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.cleaned_data[column], color="orange")
                plt.title(f"Boxplot of {column}")
                plt.xlabel(column)
                plt.show()
                
                display(f"\n=== Insights for {column} ===")
                display(f"The histogram shows the distribution of {column}, while the boxplot highlights potential outliers.")
        else:
            print("Cleaned data not available. Please clean the dataset first.")
    def univariate_analysis(self):
        """
        Conduct a non-graphical univariate analysis by computing dispersion parameters
        and provide useful interpretation.
        """
        if self.cleaned_data is not None:
            # Compute dispersion parameters
            range_values = self.cleaned_data.max(numeric_only=True) - self.cleaned_data.min(numeric_only=True)
            variance_values = self.cleaned_data.var(numeric_only=True)
            std_dev_values = self.cleaned_data.std(numeric_only=True)

            # Print results
            print("=== Non-Graphical Univariate Analysis ===")
            print("\nRange Values:")
            display(range_values)
            print("\nVariance Values:")
            display(variance_values)
            print("\nStandard Deviation Values:")
            display(std_dev_values)
            
            # Interpretation
            print("\n=== Interpretation ===")
            print("1. **Range**: Indicates the spread of values for each variable.")
            print("2. **Variance and Standard Deviation**: Measure the variability in the dataset. High values suggest greater variation, while low values indicate uniformity.")
        else:
            print("Cleaned data not available. Please clean the dataset first.")



    def bivariate_analysis(self):
        """
        Conduct bivariate analysis by exploring the relationship between 
        applications and Total Data (DL + UL).
        """
        if self.cleaned_data is not None:
            # Update the column names to match the dataset
            app_columns = [
                "Social Media DL (Bytes)", "Google DL (Bytes)", "Email DL (Bytes)", 
                "Youtube DL (Bytes)", "Netflix DL (Bytes)", "Gaming DL (Bytes)", 
                "Other DL (Bytes)"
            ]
            
            # Ensure 'Total DL (Bytes)' and 'Total UL (Bytes)' exist and calculate Total Data
            if "Total DL (Bytes)" in self.cleaned_data.columns and "Total UL (Bytes)" in self.cleaned_data.columns:
                self.cleaned_data["Total Data"] = self.cleaned_data["Total DL (Bytes)"] + self.cleaned_data["Total UL (Bytes)"]
            
            # Loop over each application and plot the bivariate analysis
            for app in app_columns:
                if app in self.cleaned_data.columns:
                    # Displaying the first few rows of the data for the current app vs Total Data
                    app_data = self.cleaned_data[[app, "Total Data"]].head()  # Display only the first few for brevity
                    print(f"Showing data for {app}:")
                    display(app_data)  # Display the first few rows
                    
                    # Plotting the scatterplot
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(
                        x=self.cleaned_data[app],
                        y=self.cleaned_data["Total Data"],
                        color="green"
                    )
                    plt.title(f"Bivariate Analysis: {app} vs Total Data")
                    plt.xlabel(app)
                    plt.ylabel("Total Data (Bytes)")
                    plt.show()
                else:
                    print(f"Column '{app}' not found in the dataset.")
        else:
            print("Cleaned data not available. Please clean the dataset first.")



    def correlation_analysis(self):
        """
        Compute the correlation matrix for selected variables and interpret the findings.
        """
        if self.cleaned_data is not None:
            # Correct column names from the dataset
            columns = [
                "Social Media DL (Bytes)", "Google DL (Bytes)", "Email DL (Bytes)", 
                "Youtube DL (Bytes)", "Netflix DL (Bytes)", "Gaming DL (Bytes)", 
                "Other DL (Bytes)"
            ]
            
            # Ensure columns exist in the dataset
            missing_columns = [col for col in columns if col not in self.cleaned_data.columns]
            if missing_columns:
                display(f"Missing columns in the dataset: {missing_columns}")
                return

            # Compute correlation matrix
            correlation_matrix = self.cleaned_data[columns].corr()
            
            # Print correlation matrix
            print("\n=== Correlation Matrix ===")
            display(correlation_matrix)
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.show()
            
            print("\n=== Insights ===")
            print("Correlation values close to 1 or -1 indicate strong positive or negative relationships.")
            print("Values close to 0 indicate weak or no correlation.")
        else:
            print("Cleaned data not available. Please clean the dataset first.")


    def dimensionality_reduction(self):
        """
        Perform Principal Component Analysis (PCA) for dimensionality reduction.
        """
        if self.cleaned_data is not None:
            # Select numeric columns and standardize
            numeric_data = self.cleaned_data.select_dtypes(include=['float64', 'int64'])
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Perform PCA
            pca = PCA()
            pca_data = pca.fit_transform(scaled_data)
            
            # Explained variance
            explained_variance = pca.explained_variance_ratio_
            print("\n=== PCA Explained Variance ===")
            for i, variance in enumerate(explained_variance):
                display(f"Principal Component {i + 1}: {variance:.2%}")
            
            # Plot explained variance
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(1, len(explained_variance) + 1),
                explained_variance.cumsum(),
                marker='o', linestyle='--', color='b'
            )
            plt.title("Explained Variance by Principal Components")
            plt.xlabel("Number of Principal Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.show()
            
            print("\n=== Insights ===")
            print("PCA helps reduce dimensions while retaining most of the variability in the data.")
            print("Select the number of components that explain a significant percentage (e.g., 95%) of the variance.")
        else:
            print("Cleaned data not available. Please clean the dataset first.")




 