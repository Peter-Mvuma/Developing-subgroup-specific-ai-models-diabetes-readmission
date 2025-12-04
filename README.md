# Developing Subgroup-Specific AI Models for Predicting 30‑Day Readmission in Patients with Diabetes

Clinical decision modeling project using subgroup-specific machine learning models to predict 30‑day hospital readmission among patients with diabetes using the **UCI “Diabetes 130‑US Hospitals (1999–2008)” dataset**.

# Goals fo the project
- Develop a **global model** and **subgroup-specific models** for predicting 30‑day readmission in diabetic patients.
- Stratify patients by **primary diagnosis**:
  - Circulatory
  - Respiratory
  - Endocrine / Nutritional / Metabolic / Immune (ENMI)
- Compare performance of subgroup models against a pooled global model.
- Identify subgroup-specific risk factors to support **clinical decision support** and targeted interventions. 

## Dataset

**Source:** UCI Machine Learning Repository – *Diabetes 130‑US Hospitals (1999–2008)*. link is https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008ource 

**Key characteristics:**

- **101,766** hospital encounters  
- **71,518** unique patients  
- **130** U.S. hospitals  
- Years **1999–2008**  
- 47 features + target outcome 

## Clinical Insights from the Dataset

- The cohort is predominantly **older adults**, skewed toward ages 60–80, with **~78% Caucasian** and **~53% female**. 
- **~80%** of admissions are emergencies, and more than half originate from the emergency room. 
- Patients readmitted within 30 days tend to have
  - Slightly **longer hospital stays**
  - **More inpatient encounters**
  - **Higher insulin use**
  - **More total medications**  
  indicating greater clinical complexity and higher treatment intensity

**Feature categories:**

- Demographics (age, race, gender)
- Admission and discharge information
- Prior utilization (inpatient, outpatient, ER visits)
- Comorbid diagnoses (ICD-based)
- Diabetes medications (e.g., metformin, insulin)
- Lab tests (HbA1c, serum glucose) 

**Target variable:**

- 30‑day readmission (binary)
  - Positive class: readmitted within 30 days
  - Negative class: readmitted after 30 days or not readmitted

## Data Preprocessing
The preprocessing pipeline ensures data quality, reduces noise, and prepares subgroup‑specific datasets. 
Main steps

- **Record selection**
  - Use **only the first encounter per patient** to avoid temporal label instability and patient‑level leakage.
    
- **Feature removal**
  - Drop identifiers (e.g., encounter ID, patient ID).
  - Remove features with **>35% missingness** (weight, medical specialty, pay code).
    
- **Missing data handling**
  - Impute remaining missing values (e.g., race, primary/secondary diagnoses) using the **mode** of each feature.
    
- **Categorical encoding and regrouping**
  - One‑hot encode multi‑category variables as needed.
  - Collapse low‑frequency categories into binary groups, such as.
    - Race: *Caucasian* vs *Other*
    - Admission type: *Emergency* vs *Other*
    - Admission source: *Emergency room* vs *Other*
    - Discharge disposition: *Home* vs *Other* 

- **Medication features**
  - Remove medications with only a single observed category.
  - Retain **metformin** and **insulin**, converting ordinal states (*up*, *down*, *steady*, *no*) to binary indicators for simplified modeling.
    
- **Diagnosis grouping**
  - Map diagnoses into:
    - Circulatory
    - Respiratory
    - Endocrine / Nutritional / Metabolic / Immune (ENMI)
    - Other
      
- **Outcome binarization**
  - Combine “>30 days” and “no readmission” into a single negative class. 

## Modeling Approach

The project implements **parallel modeling pipelines** for

- Global cohort (all qualifying encounters)
- Circulatory subgroup
- Respiratory subgroup
- ENMI subgroup 

### Train–Test Split and Resampling

- **80% / 20% train–test split** with stratification on the outcome.
- **SMOTE** applied on training data to address class imbalance.
- Scaling/standardization applied within pipelines to avoid leakage. 

### Algorithms Evaluated

For each cohort, the following models are trained and cross‑validated:
- Logistic Regression  
- Linear SVM  
- K‑Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- XGBoost  
- LightGBM 

### Evaluation Strategy

- **5‑fold cross‑validation** on the training data.  
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1‑score (macro)
  - ROC‑AUC
  - Confidence intervals for each metric from cross‑validation. 

**Model selection criterion:**  
Random Forest is chosen as the **primary model** for both global and subgroup cohorts because it offers the most balanced and robust performance across F1‑score, recall, and ROC‑AUC in cross‑validation, particularly under class imbalance. 

## Results

### Cross‑Validation (Training)

Random Forest consistently outperforms other models across cohorts. 

| Model            | Cohort      | Accuracy | Precision | Recall | F1   | ROC‑AUC |
|-----------------|------------|----------|-----------|--------|------|---------|
| Random Forest   | Global     | ~0.924   | ~0.978    | ~0.869 | ~0.920 | ~0.964 |
| Random Forest   | Circulatory| ~0.933   | ~0.981    | ~0.884 | ~0.930 | ~0.968 |
| Random Forest   | Respiratory| ~0.959   | ~0.992    | ~0.925 | ~0.957 | ~0.987 |
| Random Forest   | ENMI       | ~0.945   | ~0.976    | ~0.912 | ~0.943 | ~0.982 | 

**Key insight:** Subgroup‑specific Random Forest models, especially in the **respiratory** and **ENMI** cohorts, outperform the global model on F1‑score and ROC‑AUC, indicating that **disease‑specific stratification improves discriminative performance** in controlled evaluation. [file:1][file:2]

### Independent Test Set (20% Held‑Out)

On the untouched test set, performance across models converges. 

| Cohort      | Accuracy | Precision | Recall | F1   |
|-------------|----------|-----------|--------|------|
| Global      | 0.90     | 0.55      | 0.51   | 0.50 |
| Circulatory | 0.90     | 0.56      | 0.51   | 0.50 |
| Respiratory | 0.91     | 0.57      | 0.51   | 0.50 |
| ENMI        | 0.90     | 0.59      | 0.52   | 0.52 | 

This suggests that, while subgroup models show stronger cross‑validation metrics, their **generalization advantage on completely unseen data is modest**, highlighting ongoing challenges with class imbalance, missing data, and heterogeneity. 
 

