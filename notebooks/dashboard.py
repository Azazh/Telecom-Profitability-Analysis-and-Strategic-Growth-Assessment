import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath("../scripts"))
import db_config_dashboard

import streamlit as st
import plotly.express as px

# load dataset
# SQL query to fetch the required dataset
QUERY = "SELECT * FROM customer_satisfaction"
# Initialize the analysis class with the SQL query
data=db_config_dashboard.fetch_data(QUERY)
# Check if data is fetched successfully
# Check if data is fetched successfully
if data is not None:
    # Sidebar for Navigation
    st.sidebar.title("Dashboard Navigation")
    page = st.sidebar.radio("Go to", [
        "User Overview Analysis",
        "User Engagement Analysis",
        "Experience Analysis",
        "Satisfaction Analysis"
    ])

    # Page: User Overview Analysis
    if page == "User Overview Analysis":
        st.title("User Overview Analysis")
        st.plotly_chart(px.histogram(data, x='cluster', title="User Distribution by Cluster"))
        st.dataframe(data.describe())

    # Page: User Engagement Analysis
    elif page == "User Engagement Analysis":
        st.title("User Engagement Analysis")
        st.plotly_chart(px.scatter(data, x='userid', y='engagementscore', 
                                   title="Engagement Scores by User ID"))
        st.plotly_chart(px.box(data, y='engagementscore', title="Engagement Score Distribution"))

    # Page: Experience Analysis
    elif page == "Experience Analysis":
        st.title("Experience Analysis")
        st.plotly_chart(px.scatter(data, x='engagementscore', y='experiencescore', 
                                   title="Experience vs Engagement"))
        st.plotly_chart(px.histogram(data, x='experiencescore', title="Experience Score Distribution"))

    # Page: Satisfaction Analysis
    elif page == "Satisfaction Analysis":
        st.title("Satisfaction Analysis")
        st.plotly_chart(px.scatter(data, x='userid', y='satisfactionscore', 
                                   title="Satisfaction Scores by User ID"))
        st.plotly_chart(px.box(data, x='cluster', y='satisfactionscore', 
                               title="Satisfaction by Cluster"))
else:
    st.error("Failed to fetch data from the database. Please check your connection or query.")