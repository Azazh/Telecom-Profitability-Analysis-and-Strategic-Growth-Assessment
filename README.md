# Telecom Profitability Analysis and Strategic Growth Assessment

## 📖 **Overview**
This repository contains a comprehensive analysis and dashboard development project for  a telecommunications company. The goal is to identify business opportunities, analyze user behavior, and recommend strategies to enhance profitability and user satisfaction. The project uses data science techniques for exploratory data analysis (EDA), clustering, and regression modeling, culminating in a user-friendly dashboard for visualizing insights.

---

## 🚀 **Business Need**
TellCo's investor seeks to evaluate its customer data to determine growth opportunities and make an informed decision about acquiring or selling the company. The analysis focuses on:
- **User Overview Analysis**
- **User Engagement Analysis**
- **Experience Analytics**
- **Satisfaction Analysis**
- **Dashboard Development**

---

## 📝 **Project Tasks**

### **Task 1: User Overview Analysis**
Conducted an exploratory data analysis to understand customer demographics, device preferences, and application usage patterns.  
- Identified the **top 10 handsets** and **top 3 manufacturers**.
- Explored user behavior by aggregating metrics such as session counts, durations, download/upload data, and total data volume.
- Recommendations for marketing to target popular handsets and manufacturers.

### **Task 2: User Engagement Analysis**
Analyzed customer engagement levels to improve service quality and resource allocation:
- Aggregated engagement metrics: session frequency, duration, and total traffic.
- Clustered users into 3 groups using **K-Means Clustering** and identified top-performing customers.
- Visualized traffic distribution and application usage patterns.

### **Task 3: Experience Analytics**
Analyzed network parameters and device characteristics to evaluate user experience:
- Aggregated metrics: average TCP retransmission, RTT, throughput, and handset type.
- Performed clustering to segment users based on network experience.
- Provided actionable insights to improve service quality for different user groups.

### **Task 4: Satisfaction Analysis**
Combined engagement and experience analyses to determine customer satisfaction:
- Calculated **engagement** and **experience scores** using Euclidean distance.
- Built a regression model to predict satisfaction scores.
- Segmented customers into satisfaction clusters and exported results to a MySQL database.

### **Task 5: Dashboard Development**
Developed an interactive dashboard to visualize data insights:
- Separate pages for each task (User Overview, Engagement, Experience, and Satisfaction Analysis).
- Plots and charts for key metrics and findings.
- Designed for easy navigation and actionable insights.

---

## 🛠 **Technologies Used**
- **Programming Languages**: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Database**: Postgresql
- **Dashboard**: Streamlit
- **Clustering**: K-Means
- **Modeling**: Regression
- **Data Handling**: Data Cleaning, Outlier Treatment, Normalization
- **Visualization**: Bar Charts, Line Graphs, Correlation Matrices, PCA

---

## 📊 **Key Insights**
- The most popular handset and manufacturer can guide targeted marketing efforts.
- High engagement users contribute significantly to network traffic and should be prioritized for improved QoS.
- Segmented user experiences provide valuable insights for optimizing network parameters and enhancing satisfaction.
- Satisfaction scores reveal areas where customer retention can be improved.

---

## 📂 **Repository Structure**
```
-Telecom_Profitability_Analysis_and_Strategic_Growth_Assessment
├── .vscode/

│   └── settings.json

├── .github/

│   └── workflows

│       ├── unittests.yml

├── .gitignore

├── requirements.txt

├── README.md

├── src/

│   ├── __init__.py

├── notebooks/

│   ├── __init__.py

│   └── README.md

├── tests/

│   ├── __init__.py

└── scripts/

    ├── __init__.py

    └── README.md



---

## 🖥 **Getting Started**

### **Setup**
1. Clone this repository:
   ```bash
   git clone https://github.com/Azazh/Telecom-Profitability-Analysis-and-Strategic-Growth-Assessment.git
   cd Telecom-Profitability-Analysis-and-Strategic-Growth-Assessment.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python scripts in the `scripts/` directory for each task.
4. Launch the dashboard:
   ```bash
   navigate to notebooks folder and run
   streamlit run dashboard.py
   ```



---

### ⭐ **Acknowledgments**
Special thanks to the investor for providing this opportunity and to the 10-x team for sharing the dataset.  
