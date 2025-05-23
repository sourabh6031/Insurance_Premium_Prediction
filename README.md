# Supervised ML model - Insurance Premium Prediction App

This project aims to predict a person’s insurance premium based on various health, demographic, and medical condition-related inputs. The model is built using **XGBoost Regressor** and deployed as a **Streamlit web app**.

---

## Demo

(https://insurancepremiumprediction-sourabh6031.streamlit.app/)


## Project Structure

Insurance_Premium_Prediction/

- main.py # Streamlit app frontend
- preprocess.py # Preprocessing pipeline
- insurance_file.csv # Sample dataset
- best_xgb_grid_model.pkl # Trained XGBoost model
- scaled_scaler.pkl # Saved StandardScaler object
- requirements.txt # Python dependencies
- README.md # Project documentation


  


## Model Building

- Performed **EDA** and **feature engineering** (e.g., BMI, any_disease).
- Trained **Linear Regression, Lasso, Ridge, Random Forest, and XGBoost**.
- Used **GridSearchCV** for hyperparameter tuning.
- Final model: **XGBoost with MAE objective** for robustness to outliers.
- Key metric: **R², MAE, RMSE** (selected based on outlier presence and model goal).

---

##  Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/sourabh6031/Insurance_Premium_Prediction.git
   cd Insurance_Premium_Prediction```

2. **Create and activate virtual environment**
```
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows 
```

3. **Install dependencies**

```pip install -r requirements.txt```

4. **Run the app**
```streamlit run main.py```

```
## Input Features
Feature	                   Description
Age	                       Age in years
Diabetes	                  Binary (0/1)
BloodPressureProblems	     Binary (0/1)
AnyTransplants	            Binary (0/1)
AnyChronicDiseases	        Binary (0/1)
KnownAllergies	            Binary (0/1)
HistoryOfCancerInFamily	   Binary (0/1)
NumberOfMajorSurgeries	    0–3
Height                     In cm
Weight	                    In kg
BMI	Computed               internally (Weight/Height²)
any_disease	               Flag if any disease present
```

**Tech Stack**
- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy


**Common Errors Faced**
- ImportError from module: Caused by 0 KB or untracked file.
- ValueError: array with dim 3: Avoid double-wrapping arrays when using predict([processed_data]).
- Feature mismatch during scaling or prediction.
