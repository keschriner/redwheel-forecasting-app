# 🚲 Redwheel Inventory & Demand Forecasting Dashboard

An end-to-end, production-ready demand forecasting and decision intelligence system built for retail inventory planning. This project bridges the gap between pure machine learning forecasting and human-in-the-loop business strategy.

## 🚀 Key Features
- **Statistical Baseline:** Powered by Meta Prophet with weekly resampling (handling gaps as zeros) and promotional binary regressors (`is_promo_active`).
- **Human-in-the-Loop Override Engine:** Allows business stakeholders and inventory managers to dynamically simulate marketing pushes (like Q4 holiday bumps) and targeted SKU adjustments on top of the statistical baseline.
- **Interactive Streamlit Frontend:** A clean, decoupled web application featuring real-time KPI metrics, uncertainty interval risk bands (80% confidence intervals), and SKU-level drilldowns.
- **Robust Architecture:** Clear separation of concerns between the core data pipeline/modeling scripts and the stakeholder-facing UI.

## 🛠️ Tech Stack
- **Core Language:** Python
- **Data & Modeling:** Pandas, NumPy, Meta Prophet
- **UI Framework:** Streamlit
- **Visualization:** Matplotlib

## 📂 Project Structure

redwheel-forecasting-app/
│
├── data/
│   └── redwheel_final_sales_forecast.csv  # Processed master forecast table
│
├── src/
│   ├── init.py                      # Package initialization & clean imports
│   └── pipeline.py                      # Data harmonization & Prophet training loop
│
├── app.py                               # Main Streamlit dashboard application
├── requirements.txt                     # Package dependencies
└── README.md                            # Project documentation

## 🏃‍♂️ How to Run Locally

1. **Clone the repository:**

git clone https://github.com/your-username/redwheel-forecasting-app.git
cd redwheel-forecasting-app


2. **Install dependencies:**

pip install -r requirements.txt


3. **Launch the Streamlit app:**

streamlit run app.py
