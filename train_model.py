# =====================================
# Water Potability - Full Training File
# =====================================

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv('./data/water_potability.csv')

# -----------------------------
# 2. Missing Value Handling (EDA decisions)
# -----------------------------
data['ph'] = data['ph'].fillna(data.groupby('Potability')['ph'].transform('mean'))
data['Sulfate'] = data['Sulfate'].fillna(
    data.groupby('Potability')['Sulfate'].transform('median')
)
data['Trihalomethanes'] = data['Trihalomethanes'].fillna(
    data.groupby('Potability')['Trihalomethanes'].transform('mean')
)

# -----------------------------
# 3. Feature Engineering
# -----------------------------
data['ph_squared'] = data['ph'] ** 2
data['Sulfate_squared'] = data['Sulfate'] ** 2
data['Chloramines_squared'] = data['Chloramines'] ** 2

# -----------------------------
# 4. Outlier Handling (5th–95th percentile capping)
# -----------------------------
def cap_outliers(df, column):
    lower = df[column].quantile(0.05)
    upper = df[column].quantile(0.95)
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])
    return df

numeric_cols = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Conductivity',
    'Organic_carbon', 'Sulfate', 'Trihalomethanes', 'Turbidity',
    'ph_squared', 'Sulfate_squared', 'Chloramines_squared'
]

for col in numeric_cols:
    data = cap_outliers(data, col)

# -----------------------------
# 5. Feature Selection
# -----------------------------
features = numeric_cols
X = data[features]
y = data['Potability']

# -----------------------------
# 6. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 7. Pipeline (Scaling + SMOTE + Model)
# -----------------------------
pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

# -----------------------------
# 8. Hyperparameter Tuning
# -----------------------------
param_distributions = {
    'n_estimators': list(np.arange(100, 501, 50)),      
    'max_depth': [None] + list(np.arange(5, 51, 5)),    
    'min_samples_split': list(np.arange(2, 21, 2)),   
    'min_samples_leaf': list(np.arange(1, 11)),        
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

random_search_rf = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,                 
    scoring='roc_auc',        
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("🚀 Training model with full EDA pipeline...")
random_search_rf.fit(X_train, y_train)



# -----------------------------
# 9. Save best pipeline
# -----------------------------
best_model = random_search_rf.best_estimator_

from sklearn.metrics import roc_auc_score

y_test_pred = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_pred)

print("🧪 Test ROC-AUC:", test_auc)


print("✅ Best CV ROC-AUC:", random_search_rf.best_score_)
print("✅ Best Parameters:", random_search_rf.best_params_)

joblib.dump(best_model, './models/water_pipeline.joblib')

print("💾 Full pipeline saved successfully!")
