from sklearn.datasets import fetch_openml
import pandas as pd
# Fetch dataset
data = fetch_openml(name="student_depression_dataset", as_frame=True)
# Convert to DataFrame
df = data.frame
df.to_csv("student_depression_dataset.csv", index=False)

df
import os
print(os.getcwd())
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\malle\student_depression_dataset.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# Drop ID column 
df.drop(columns=['id'], inplace=True)

# Numerical columns → fill with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values after cleaning:\n", df.isnull().sum())
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    
# SPLIT FEATURES & TARGET
X = df.drop(columns=['Depression'])   
y = df['Depression'] 
                
# TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# FEATURE SCALING
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# INITIALIZE MODELS
dt_model = DecisionTreeClassifier(random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

# ---- Decision Tree ----
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", dt_acc)
print(classification_report(y_test, dt_pred))

# ---- KNN ----
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

print("\nKNN Accuracy:", knn_acc)
print(classification_report(y_test, knn_pred))

# ---- Random Forest ----
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_acc)
print(classification_report(y_test, rf_pred))

results = pd.DataFrame({
    'Model': ['Decision Tree', 'KNN', 'Random Forest'],
    'Accuracy': [dt_acc, knn_acc, rf_acc]
})

print("\nModel Comparison:\n")
print(results)

# BEST MODEL SELECTION
best_model = results.loc[results['Accuracy'].idxmax()]
print("\nBest Performing Model:")
print(best_model)
import matplotlib.pyplot as plt

# Model names and accuracies
models = ['Decision Tree', 'KNN', 'Random Forest']
accuracies = [dt_acc, knn_acc, rf_acc]
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Confusion matrix
cm = confusion_matrix(y_test, rf_pred)

# 1. Plot confusion matrix
plt.figure()
plt.imshow(cm)
plt.colorbar()
plt.xticks([0, 1], ['No Depression', 'Depression'])
plt.yticks([0, 1], ['No Depression', 'Depression'])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Random Forest')
plt.show()

#2
plt.figure()
plt.boxplot(
    [df[df['Depression']==0]['Age'],
     df[df['Depression']==1]['Age']],
    labels=['No Depression', 'Depression']
)
plt.xlabel('Class')
plt.ylabel('Age')
plt.title('Age vs Depression (Box Plot)')
plt.show()

#3
import numpy as np
import matplotlib.pyplot as plt
corr = df.corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Correlation Heatmap')
plt.show()

#4
models = ['Decision Tree', 'KNN', 'Random Forest']
accuracies = [dt_acc, knn_acc, rf_acc]
plt.figure()
plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of ML Models')
plt.ylim(0,1)
plt.show()

#5
from sklearn.metrics import roc_curve, auc
rf_prob = rf_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, rf_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.show()

#6
import numpy as np
importances = rf_model.feature_importances_
features = X.columns
idx = np.argsort(importances)[::-1][:10]
plt.figure()
plt.bar(range(len(idx)), importances[idx])
plt.xticks(range(len(idx)), features[idx], rotation=90)
plt.ylabel('Importance Score')
plt.title('Top 10 Important Features')
plt.show()

