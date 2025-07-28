# Clinical Trial Analysis with Survival Modeling & Machine Learning

This project simulates the end-to-end data analysis pipeline for a **Phase III Clinical Trial** assessing the effectiveness and safety of a novel oncology drug. It combines traditional statistical techniques with machine learning to evaluate patient outcomes and model risk factors for early discontinuation.

##  Project Purpose

In clinical trials, especially in oncology, understanding both **time-to-event outcomes** and **risk factors** is critical. This project replicates key aspects of what a Data Analyst or Biostatistician would do in a real-world trial:

- Perform survival analysis using Kaplan-Meier estimators
- Predict early discontinuation using a machine learning model
- Visualize and interpret clinical data using Python

##  Key Features

 1. **Descriptive Analytics**
- Summary statistics of clinical features
- Age, biomarker levels, ECOG performance scores

 2. **Survival Analysis**
- Kaplan-Meier estimator applied to time-to-event data
- Graphical plot showing survival probability over time

 3. **Predictive Modeling**
- Random Forest Classifier to predict early discontinuation risk
- ROC AUC evaluation and classification metrics
- Confusion matrix heatmap

## Dataset Used

A synthetic dataset `clinical_trial_data.csv` (500 patients) includes:

| Column                | Description                                      |
|-----------------------|--------------------------------------------------|
| `age`                 | Patient's age                                    |
| `baseline_lab1`       | Lab value at baseline (e.g., WBC count)          |
| `biomarker_level`     | Biomarker of interest (continuous variable)      |
| `ecog_score`          | ECOG Performance Status (0 = best, 2 = worst)    |
| `time_to_event`       | Time (days) to disease progression or censoring  |
| `event_occurred`      | Whether a progression or death occurred (1/0)    |
| `early_discontinuation` | Whether the patient dropped out early (1/0)  |

---

## ML & Statistical Tools Used

- **Lifelines** – for Kaplan-Meier survival curves
- **Scikit-learn** – for machine learning classification (Random Forest)
- **Pandas / NumPy** – for data manipulation
- **Matplotlib / Seaborn** – for visualization


## 📁 File Structure
clinical-trial-analysis/
│
├── clinical_trial_analysis.py 
├── clinical_trial_data.csv 
├── km_curve.png 
├── confusion_matrix.png 
└── README.md 

Output
•	Terminal: Summary statistics, model evaluation metrics, ROC AUC score
•	Images: km_curve.png, confusion_matrix.png will be saved to your project directory

Use Cases
•	Showcase skills for roles in pharma analytics, clinical data management, biostatistics, or healthcare data science
•	Demonstrate experience in real-world domains like oncology, trial monitoring, and patient risk modelling

Future Enhancements
•	Add Cox Proportional Hazards modeling
•	Support multi-arm trial analysis (treatment vs placebo)
•	Integrate with Streamlit or Dash for an interactive dashboard
•	Export results to Excel/PDF for regulatory reports

License
MIT License – Feel free to fork, use, and modify this project.
Author
Divyabarathi
📧 bharathi.divya87@gmail.com


