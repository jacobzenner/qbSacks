# Predicting NFL Sacks with Logistic Regression and XGBoost  
### Author: Jacob Zenner

This project explores the use of **logistic regression** and **XGBoost** to predict the likelihood of a quarterback sack on a passing play using raw NFL play-by-play data from the **2019–2023 seasons**.

---

## Objective
To determine what conditions make a quarterback sack more likely on a given NFL play, using structured play-by-play data and machine learning models.

---

## Tools & Libraries
- [nfl_data_py](https://pypi.org/project/nfl-data-py/)
- `pandas`, `numpy` – Data wrangling
- `matplotlib`, `seaborn` – Visualizations
- `sklearn` – Logistic regression, data splitting, metrics
- `xgboost` – Gradient-boosted classifier

---

## Workflow Summary

1. **Data Acquisition**  
   NFL play-by-play data imported using `nfl_data_py` for 2019–2023.

2. **Feature Engineering**  
   - Filtered to passing plays only  
   - Created binary `likely_pass` indicator  
   - Selected relevant features (e.g., `down`, `defenders_in_box`, `number_of_pass_rushers`)  
   - One-hot encoded categorical variables like `down`  

3. **Data Cleaning**  
   Removed rows with missing values to ensure model integrity.

4. **Train/Test Split**  
   Used `StratifiedShuffleSplit` to maintain class balance across training and test sets.

5. **Model Training & Evaluation**  
   - **Logistic Regression**  
     - Evaluated with **accuracy** and **Brier score**  
   - **XGBoost Classifier**  
     - Evaluated with **Brier score**  
     - Feature importance visualized

---

## 🔍 Key Insights
- Higher sack frequency observed on **3rd downs** and in **likely pass situations**
- **Number of pass rushers** and **defenders in the box** correlate with sack probability
- XGBoost provided improved predictive performance over logistic regression

---

## 📈 Sample Output
- Accuracy: *Displayed in console*
- Brier Score: *Displayed in console*
- Feature importance graph for XGBoost model
