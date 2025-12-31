import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
# def evaluate_models_cv(x, y, models, cv_splits=5):
#     """
#     Evaluate multiple models using Stratified K-Fold CV with SMOTE.
#     Returns a dict of classification reports for each model.
#     """
#     reports = {}
#     cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

#     for name, model in models.items():
#         # Build pipeline: SMOTE + model
#         pipeline = Pipeline([
#             ('smote', SMOTE(random_state=42)),
#             (name.lower().replace(" ", "_"), model)
#         ])

#         # Cross-validated predictions
#         y_pred = cross_val_predict(pipeline, x, y, cv=cv)

#         # Classification report
#         report = classification_report(y, y_pred, output_dict=True)
#         reports[name] = report

#     return reports

    
def evaluate_models(x_train, y_train, x_test, y_test, models):
    results = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

    # ✅ return AFTER the loop
    df_results = pd.DataFrame(results).set_index("Model")
    return df_results

    



           

        
    
    
    