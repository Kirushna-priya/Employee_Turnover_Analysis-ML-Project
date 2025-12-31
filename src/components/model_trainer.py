import os
import sys
from dataclasses import dataclass 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter= 10000),
                "Random Forest Classifier": RandomForestClassifier(random_state = 42, n_estimators=100,n_jobs=-1),
                "Gradient Boosting classifier": GradientBoostingClassifier(random_state = 42,n_estimators=100, learning_rate=0.1)
            }

            model_report = evaluate_models(x_train, y_train, x_test, y_test, models)
            print(model_report)  # Debug: see all metrics

            best_model_name = model_report["Accuracy"].idxmax()
            best_model_score = model_report.loc[best_model_name, "Accuracy"]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
               raise CustomException(f"No best model found (score={best_model_score:.2f})")

            logging.info(f"Best found model: {best_model_name} with accuracy {best_model_score:.2f}")

        

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
        
        except CustomException as ce:
            raise ce


        except Exception as e:
            raise CustomException(e, sys)


# #  Evaluate with 5-fold Stratified CV
#             cv_reports = evaluate_models_cv(x_train, y_train, models)

#             # Pick best model by mean F1-score (or accuracy)
#             best_model_name = max(cv_reports, key=lambda name: cv_reports[name]['accuracy'])
#             best_model = models[best_model_name]

#             logging.info(f"Best model from CV: {best_model_name}")

#             # Fit best model on full training set
#             best_model.fit(x_train, y_train)

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted = best_model.predict(x_test)
#             accuracy = accuracy_score(y_test, predicted)
#             return accuracy

#         except Exception as e:
#             raise CustomException(e, sys)
