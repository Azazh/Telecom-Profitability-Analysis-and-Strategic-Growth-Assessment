import pandas as pd
import matplotlib.pyplot as plt

import db_config

class UserOverviewAnalysis:
    def __init__(self, query):
        """
        Initialize the class with a SQL query to fetch data.
        :param query: The SQL query to fetch the dataset.
        """
        # Load the dataset from the database
        self.data = query

    def top_handsets(self, top_n=10):
        """Identify the top N handsets used by customers."""
        top_handsets = self.data['Handset Type'].value_counts().head(top_n)
        return top_handsets

    def top_handset_manufacturers(self, top_n=3):
        """Identify the top N handset manufacturers."""
        top_manufacturers = self.data['Handset Manufacturer'].value_counts().head(top_n)
        return top_manufacturers

    def top_handsets_per_manufacturer(self, manufacturers, top_n=5):
        """Identify the top N handsets per given manufacturers."""
        result = {}
        for manufacturer in manufacturers:
            top_handsets = (
                self.data[self.data['Handset Manufacturer'] == manufacturer]['Handset Type']
                .value_counts()
                .head(top_n)
            )
            result[manufacturer] = top_handsets
        return result