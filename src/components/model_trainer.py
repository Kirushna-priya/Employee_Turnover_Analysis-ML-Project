import os
import sys
from dataclasses import dataclass 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
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

    def tune_hyperparameters(self, model, param_grid, x_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV with Stratified K-Fold.
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(x_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


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
            
            param_grids = {
                  "Logistic Regression": {
                  "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear", "lbfgs"]
                    },
                    "Random Forest Classifier": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                            },
                    "Gradient Boosting classifier": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5]
                    }
                    }

            best_models = {}
            
            for name, model in models.items():
                 best_estimator, best_params, best_score = self.tune_hyperparameters(
                   model, param_grids[name], x_train, y_train
                   )
                
                 logging.info(f"{name} best params: {best_params}, CV score: {best_score:.4f}")
                
                 best_models[name] = best_estimator

            # Evaluate all model

            model_report = evaluate_models(x_train, y_train, x_test, y_test, models=models)
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
        
    def evaluate_on_test(self, best_models, x_test, y_test):
        results = {}
        for name, model in best_models.items():
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
                    
            logging.info(f"{name} test accuracy: {acc:.4f}")
                    
        results[name] = acc
        return results

        
        



