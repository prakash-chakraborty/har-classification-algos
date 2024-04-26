import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Load the datasets
train_data_path = './dataset/train.csv'
test_data_path = './dataset/test.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Preprocess the data
# Drop 'subject' column as it's not a feature
train_df = train_df.drop(['subject'], axis=1)
test_df = test_df.drop(['subject'], axis=1)

# Separate features and labels
X_train = train_df.drop('Activity', axis=1)
y_train = train_df['Activity']
X_test = test_df.drop('Activity', axis=1)
y_test = test_df['Activity']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM with hyperparameter tuning
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':('scale', 'auto')}
svc = SVC(probability=True)  # Enable probability estimate for AUC calculation
clf = GridSearchCV(svc, parameters)
clf.fit(X_train_scaled, y_train)
best_svm_model = clf.best_estimator_

# Evaluate the best model
y_pred = best_svm_model.predict(X_test_scaled)

# Calculate the different metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


# AUC calculation requires binary label indicators
y_test_binarized = label_binarize(y_test, classes=list(set(y_train)))  # Adjust based on the actual labels
y_score = best_svm_model.predict_proba(X_test_scaled)
auc_score = roc_auc_score(y_test_binarized, y_score, multi_class='ovr', average='weighted')

#Print Metrics
print("Best SVM Model:", clf.best_params_)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC Score:", auc_score)