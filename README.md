# Football Scouting Predictor

## Project Overview

The Football Scouting Predictor is a data analytics and machine learning project that analyses historical FIFA player data to identify high-potential youth football players.

The project demonstrates an end-to-end analytics workflow including data exploration, preprocessing, feature engineering, model training, prediction scoring, and an interactive Streamlit dashboard for scouting analysis.

Developed as part of Capstone analytics skill development.


## Objectives

- Analyse structured football player datasets
- Identify high-growth youth prospects
- Build an interpretable growth prediction model
- Create a scouting score metric
- Provide an interactive scouting dashboard


## Key Features

- Data cleaning and preprocessing pipeline
- Growth Potential metric (Potential − Overall)
- Machine learning growth prediction model
- Feature pipeline saved with joblib
- Youth prospect filtering
- Streamlit scouting dashboard
- Exportable prospect shortlist


## Dataset Files

Located in `/Data` folder:

- `CompleteDataset.csv` — original dataset
- `CleanedDataset.csv` — processed dataset
- `YouthTalentProspects.csv` — filtered youth shortlist output

FIFA ratings are synthetic but internally consistent and suitable for prototyping prediction systems.


## Model & Pipeline

Located in `/models` folder:

- `growth_model.joblib` — trained ML model
- `growth_features.joblib` — feature pipeline

These files allow predictions without retraining each time.


## Technologies Used

- Python
- pandas
- scikit-learn
- Streamlit
- joblib
- matplotlib / plotting libraries


## How to Run

- pip install -r requirements.txt
- streamlit run app.py


## Dataset Source

This project uses a publicly available FIFA player dataset obtained from Kaggle for educational and analytics prototyping purposes. All analysis and modelling work in this repository was developed independently on top of that dataset.
