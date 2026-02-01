EMPLOYEE TURNOVER ANALYSIS

1.PROBLEM OVERVIEW:

Portobello Tech is an app innovator who has devised an intelligent way of predicting employee turnover within the company. It periodically evaluates employees' work details, including the number of projects they worked on, average monthly working hours, time spent in the company, promotions in the last five years, and salary level.

Data from prior evaluations shows the employees’ satisfaction in the workplace. The data could be used to identify patterns in work style and their interest in continuing to work for the company. 

The HR Department owns the data and uses it to predict employee turnover. Employee turnover refers to the total number of workers who leave a company over time.

 1.1 PROBLEM STATEMENT:

This project understands how the employee turnover is affected by different independent variables like satisfaction level, last evaluation, number of projects, average montly working hours, time spend in the company, Work accident occurence,promotion in last 5years,	department and salary.

2. DATASET COLLECTION:
Data source: https://www.kaggle.com/liujiaqi/hr-comma-sepcsv

The data consist of 10 columns and 14999 rows.

2.2 DATASET INFORMATION:

a.satisfaction_level:Satisfaction level at the job of an employee

b.last_evaluation:Rating between 0 and 1, received by an employee at his last evaluation

c.number_project:The number of projects an employee is involved in

d.average_montly_hours:Average number of hours in a month spent by an employee at the office

e.time_spend_company:Number of years spent in the company

f.Work_accident	: 0 - no accident during employee stay, 1 - accident during employee stay

g.left: 0 indicates an employee stays with the company and 1 indicates an employee left the company

h.promotion_last_5years: Number of promotions in his stay

i.Department:Department to which an employee belongs to

j.salary: Salary in USD

 
During the development and deployment of the Employee Turnover Analysis Flask Application, several important lessons were learned:

1. Initial Success
- The Flask website worked well locally with the trained model and preprocessing pipeline.
- Confidence was high, so a Dockerfile was written to containerize the application.

2. First Major Error
- When running the Docker container, the application threw:
   ModuleNotFoundError: No module named 'sklearn.ensemble._gb_losses'
- This error was caused by a scikit-learn version mismatch:
- Old pickle files (model.pkl, preprocessor.pkl) were created with an older scikit-learn version.
- Newer versions removed internal modules like _gb_losses.

3. Options Considered
- Quick Fix: Downgrade scikit-learn to the older version (compatible with the pickle files).
- Long-Term Fix: Upgrade scikit-learn, delete old pickle files, and retrain the model for future stability.

4. Chosen Path
- Opted for the long-term fix:
- Deleted old pickle files.
- Installed newer versions of scikit-learn.
- Noticed dependency warnings: numba and opencv-python were incompatible with the latest NumPy.
- Upgraded those libraries as well to ensure compatibility.

5. Retraining
- Retrained the ML model and preprocessing pipeline.
- New pickle files (model.pkl, preprocessor.pkl) were generated with the updated environment.

6. Final Outcome
- The Flask website now runs successfully inside Docker with the new setup.
- The project is future-proofed with:
- Updated dependencies.
- Reproducible training pipeline.
- Clear separation of preprocessing and model pickle files.

🚀 Key Takeaways
- Always pin dependency versions in requirements.txt or environment.yml to avoid mismatches.
- Avoid relying on private scikit-learn modules (_gb_losses) — stick to public APIs.
- When upgrading libraries, check for downstream compatibility (NumPy, numba, OpenCV, etc.).
- Retraining models after major library upgrades ensures long-term stability and reproducibility.








