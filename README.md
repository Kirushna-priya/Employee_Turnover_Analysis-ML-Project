# Employee Turnover Analysis & Prediction

A machine learning project that analyzes HR data to **predict employee turnover** and provides insights into what factors most influence employees leaving a company.  
Built with Python, scikit-learn, and deployed as a Flask app with Docker support.

---

## 🚀 Project Overview

Employee turnover is costly — both financially and in lost institutional knowledge.  
This project predicts whether an employee will leave using historical HR data and a classification model.

**Business value:**  
- Helps HR teams identify at-risk employees  
- Supports data-driven retention strategies  
- Improves employee engagement and reduces cost of rehiring

---

## 📦 Features

✔ Data ingestion and preprocessing  
✔ Exploratory Data Analysis (EDA)  
✔ Feature engineering  
✔ Model training and evaluation  
✔ Prediction API served via Flask  
✔ Containerized deployment (Docker)  
✔ Clear separation between training and inference logic

---

## 📁 Repository Structure
```text

├── artifacts/ # Model artifacts and pickle files
├── notebook/ # Exploratory analyses and experimentation
├── src/
│ ├── data_processing.py # Data cleaning & feature pipeline
│ ├── model.py # Model training logic
│ ├── predict.py # Inference utilities
├── templates/ # Web UI templates for Flask app
├── app.py # Flask application
├── requirements.txt # Dependencies
├── Dockerfile # Container config
├── README.md
└── setup.py
```
## 📊 Dataset

Data source: https://www.kaggle.com/liujiaqi/hr-comma-sepcsv

The data consist of 10 columns and 14999 rows.

## Dataset Details

- satisfaction_level:Satisfaction level at the job of an employee
- last_evaluation:Rating between 0 and 1, received by an employee at his last evaluation
- number_project:The number of projects an employee is involved in
- average_montly_hours:Average number of hours in a month spent by an employee at the office
- time_spend_company:Number of years spent in the company
- Work_accident	: 0 - no accident during employee stay, 1 - accident during employee stay
- left: 0 indicates an employee stays with the company and 1 indicates an employee left the company
- promotion_last_5years: Number of promotions in his stay
- Department:Department to which an employee belongs to
- salary: Salary in USD

## 📈 Model Training

I trained a classification model (e.g., Random Forest Classifier / Gradient Boosting / Logistic Regression) to predict employee attrition.  
Evaluation metrics include:

- **Accuracy**
- **Precision / Recall**
- **F1 score**
- **Confusion Matrix**
- **ROC–AUC**

  **MODEL 1: Logistic Regression + 5 fold CV**

```text
    precision    recall  f1-score   support

           0       0.96      0.77      0.85     10000
           1       0.41      0.82      0.55      1991

    accuracy                           0.78     11991
   macro avg       0.68      0.79      0.70     11991
weighted avg       0.86      0.78      0.80     11991
```
**MODEL 2: Random Forest Classifier + 5 Fold CV**
```text
    precision    recall  f1-score   support

           0       0.98      1.00      0.99     10000
           1       0.98      0.91      0.95      1991

    accuracy                           0.98     11991
   macro avg       0.98      0.96      0.97     11991
weighted avg       0.98      0.98      0.98     11991
```
  **MODEL 3: Gradient Boosting Classifier + 5 Fold CV**
```text
    precision    recall  f1-score   support

           0       0.99      0.98      0.98     10000
           1       0.91      0.93      0.92      1991

    accuracy                           0.97     11991
   macro avg       0.95      0.95      0.95     11991
weighted avg       0.97      0.97      0.97     11991
```
-  Clearly the performance of Random Forest Classifier model and Gradient Boosting Classifier are the best with recall value for class 1 as 0.91 and 0.93 respectively.

- According to the problem statement, **recall** is the most important metric because a false negative entry becomes costly for the company.

## Model Prediction
 - The model predicts if the person will stay or leave of company based on Satisfaction level, number of projects, average monthly hours, time spend in the company, department and salary. 

## 🔧 Local Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Kirushna-priya/Employee_Turnover_Analysis-ML-Project.git
   ```
2. **Create & activate virtual environment:**
   ```python3 -m venv venv
    source venv/bin/activate
   ```
3. **Install dependencies:**
   ```pip install -r requirements.txt```

4. **Train the model:**
   ```python src/model.py```

5. **Run the app locally:**
   ```python app.py```

6. **Visit the UI:**
   ```http://localhost:5000```

7. **🐳 Docker Deployment**
   ```docker build -t employee-turnover .
      docker run -p 5000:5000 employee-turnover
   ```
## Challenges

During the development and deployment of the Employee Turnover Analysis Flask Application, several important lessons were learned:

**1. Initial Success**
- The Flask website worked well locally with the trained model and preprocessing pipeline.
- Confidence was high, so a Dockerfile was written to containerize the application.

**2. First Major Error**
- When running the Docker container, the application threw:
   ModuleNotFoundError: No module named 'sklearn.ensemble._gb_losses'
- This error was caused by a scikit-learn version mismatch:
- Old pickle files (model.pkl, preprocessor.pkl) were created with an older scikit-learn version.
- Newer versions removed internal modules like _gb_losses.

**3. Options Considered**
- Quick Fix: Downgrade scikit-learn to the older version (compatible with the pickle files).
- Long-Term Fix: Upgrade scikit-learn, delete old pickle files, and retrain the model for future stability.

**4. Chosen Path**
- Opted for the long-term fix:
- Deleted old pickle files.
- Installed newer versions of scikit-learn.
- Noticed dependency warnings: numba and opencv-python were incompatible with the latest NumPy.
- Upgraded those libraries as well to ensure compatibility.

**5. Retraining**
- Retrained the ML model and preprocessing pipeline.
- New pickle files (model.pkl, preprocessor.pkl) were generated with the updated environment.

**6. Final Outcome**
- The Flask website now runs successfully inside Docker with the new setup.
- The project is future-proofed with:
- Updated dependencies.
- Reproducible training pipeline.
- Clear separation of preprocessing and model pickle files.

##🚀 Key Takeaways

- Always pin dependency versions in requirements.txt or environment.yml to avoid mismatches.
- Avoid relying on private scikit-learn modules (_gb_losses) — stick to public APIs.
- When upgrading libraries, check for downstream compatibility (NumPy, numba, OpenCV, etc.).
- Retraining models after major library upgrades ensures long-term stability and reproducibility.








