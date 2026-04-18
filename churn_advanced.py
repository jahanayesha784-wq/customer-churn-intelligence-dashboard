import os
import joblib
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore")

# Create folders if missing
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

# Drop customerID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Convert target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Split input and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Detect column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric columns:", numeric_features)
print("Categorical columns:", categorical_features)

# Numeric preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Full preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

# Hyperparameter tuning
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

print("\nTraining with GridSearchCV...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# Cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="roc_auc")
print("\nCross-validation ROC-AUC scores:", cv_scores)
print("Mean CV ROC-AUC:", cv_scores.mean())

# Predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nTest Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc)

# Save results to text file
with open("models/results.txt", "w") as f:
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Mean CV ROC-AUC: {cv_scores.mean():.4f}\n")
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curve.png")
plt.close()

# Feature importance
ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
encoded_cat_features = ohe.get_feature_names_out(categorical_features)
all_features = numeric_features + list(encoded_cat_features)

importances = best_model.named_steps["classifier"].feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance_df.head(10),
    x="Importance",
    y="Feature"
)
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

# Save model
joblib.dump(best_model, "models/churn_advanced_model.pkl")
print("\nModel saved to models/churn_advanced_model.pkl")
print("Plots saved to plots/")
print("Results saved to models/results.txt")
