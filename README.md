# Gurgaon Real Estate Insights 🏡

An AI-powered real estate analytics web app focused on Gurgaon's property market.
Built with Machine Learning, Data Visualization, and Streamlit.

---

## Features

- **Market Analysis** — Explore locality-wise trends, visualize property data, and identify the best investment zones in Gurgaon
- **Price Predictor** — Enter property details and get an AI-based price estimate instantly
- **Apartment Recommender** — Get personalized apartment suggestions based on budget, safety, and amenities

---

## Tech Stack

- **Frontend/UI:** Streamlit
- **Language:** Python
- **ML & Data:** scikit-learn, pandas, numpy
- **Visualization:** matplotlib / seaborn / plotly
- **Notebooks:** Jupyter

---

## Project Structure
Captone_project/

├── data/               # Raw and processed property datasets

├── model/              # Trained ML models

├── notebooks/          # EDA and model training notebooks

├── pages/              # Streamlit multi-page modules

│   ├── market_analysis.py

│   ├── price_predictor.py

│   └── recommendations.py

├── home.py             # App entry point

└── .gitignore

---

## Installation

```bash
git clone https://github.com/Abhinandan2006/Captone_project.git
cd Captone_project
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run home.py
```

Then open `http://localhost:8501` in your browser and use the sidebar to navigate between modules.

---

## Dataset

Property data sourced from real listings in Gurgaon, India. Raw data is in the `data/` folder and preprocessing steps are documented in the notebooks.
---
## License
This project is for learning and educational purposes.
