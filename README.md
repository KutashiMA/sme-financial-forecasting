# UK SME Financial Data: Time Series Forecasting

This project analyzes and forecasts key financial metrics for UK Small and Medium-sized Enterprises (SMEs) using time series analysis. The core of the project involves cleaning and transforming a raw dataset, performing exploratory data analysis (EDA) to uncover trends, and building two different forecasting models: a classical statistical **ARIMA** model and a deep learning **LSTM** network.

## Project Objective

The main goal is to predict future trends in SME financial performance (specifically Turnover, Profit, and Corporation Tax) based on historical data. This project demonstrates an end-to-end data science workflow, from data ingestion and cleaning to modeling and evaluation.

## Technologies Used

*   **Data Manipulation:** Pandas, NumPy
*   **Data Visualization:** Matplotlib, Seaborn
*   **Statistical Modeling:** Statsmodels (for ARIMA)
*   **Deep Learning:** TensorFlow (with Keras for LSTM)
*   **Machine Learning Utilities:** Scikit-learn
*   **Development Environment:** Jupyter Notebook

## Project Structure

```
sme-financial-forecasting/
├── data/
│   └── FINANCIAL DATA of UK SMEs.csv
├── notebooks/
│   └── sme_financial_analysis.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run This Project

To replicate the analysis and run the models, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KutashiMA/sme-financial-forecasting.git
    cd sme-financial-forecasting
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate it (on Windows)
    venv\Scripts\activate

    # Activate it (on macOS/Linux)
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter and run the notebook:**
    Open the `sme_financial_analysis.ipynb` file located in the `notebooks/` directory to see the complete analysis, code, and visualizations.

## Analysis and Modeling Workflow

The project follows a structured data science workflow documented within the notebook:

1.  **Data Loading and Preprocessing:**
    *   The raw CSV data is loaded into a Pandas DataFrame.
    *   The dataset is transformed from a wide format (years as columns) to a long format suitable for time series analysis, with a proper `Date` column.
    *   Data types are corrected, and missing values are handled.

2.  **Exploratory Data Analysis (EDA):**
    *   Visualizations are used to explore the trends, seasonality, and distribution of the financial metrics over time.
    *   A correlation heatmap is generated to understand the relationships between Turnover, Profit, and Tax.

3.  **Time Series Forecasting:**
    *   The data is split into training and testing sets for model evaluation.
    *   **ARIMA Model:** A traditional Autoregressive Integrated Moving Average model is built to establish a statistical baseline for forecasting.
    *   **LSTM Network:** A Long Short-Term Memory neural network, a more complex deep learning model, is implemented to capture non-linear patterns in the time series data.

4.  **Model Evaluation:**
    *   Both models are evaluated on the test set using standard regression metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
    *   The performance of the ARIMA and LSTM models is compared to determine which approach is more effective for this dataset.

## Author

*   **Muhammed Kutashi**
    *   GitHub: [@KutashiMA](https://github.com/KutashiMA)
