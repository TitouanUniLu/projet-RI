# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
import os
import codecs
import re

# Chargement des données:
def load_pres(fname):
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else: 
            alllabs.append(1)
        alltxts.append(txt)
    return alltxts,alllabs


# +
fname = "../datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = load_pres(fname)

print(f'{len(alltxts)} phrases')
print('Chirac == label 1 et Mitterand == label -1')
print(f'{alltxts[0]} -> classe: {alllabs[0]}')
print(f'{alltxts[11]} -> classe: {alllabs[11]}')
print(f'Chirac: {np.sum(np.array(alllabs) == 1)} phrases - Mitterand: {np.sum(np.array(alllabs) == -1)} phrases')
print(f'on remarque que Chirac a parlé {np.round(49890/7523)} fois plus que Mitterand')

# +
from scipy.ndimage import gaussian_filter

def gaussian_pred_smoothing(model, X, sigma):
        # Définition du noyau de filtre gaussien
        pred =  model.predict_proba(X)
        smoothed_pred = gaussian_filter(pred, sigma)
        return smoothed_pred


# +
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(use_idf=True, norm='l2', smooth_idf=True, lowercase=False)
X = tfidf.fit_transform(alltxts)

X_train, X_test, y_train, y_test = train_test_split(X, alllabs, test_size=0.20, random_state=12)

# +
from imblearn.over_sampling import RandomOverSampler

oversampler = RandomOverSampler(sampling_strategy='minority', random_state=0)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# +
num_cores = os.cpu_count()

print("Number of CPU cores available:", num_cores)

cores = int(0.2*num_cores)
print(cores)
cores = 2

# +
from imblearn.pipeline import make_pipeline

pipeline = make_pipeline(
    RandomOverSampler(sampling_strategy='minority', random_state=0),
    LogisticRegressionCV(cv=5, scoring='f1', n_jobs=cores, verbose=1, max_iter=1000)
)

param_grid = {
    'logisticregressioncv__Cs': [0.1, 1, 10],
    'logisticregressioncv__penalty': ['l1', 'l2'],
    'logisticregressioncv__solver': ['liblinear', 'saga'],
    'logisticregressioncv__fit_intercept': [True],
    'logisticregressioncv__max_iter': [1000, 2000],
    'logisticregressioncv__class_weight': ['balanced', {1: 0.7, -1: 0.3}, {1: 1, -1: 10}] #
}

# Create GridSearchCV object with scoring='f1'
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='f1', n_jobs=cores, verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

log_reg_classifier = grid_search.best_estimator_

if False: #save code just in case
    log_reg_classifier = make_pipeline(
        oversampler,
        LogisticRegressionCV(
            cv=5, 
            scoring='f1', 
            random_state=0, 
            n_jobs=-1,
            verbose=3,
            max_iter=1000,
            class_weight='balanced'
        )
    )

    log_reg_classifier.fit(X_train_resampled, y_train_resampled)

# +
import joblib 

model_filename = "best_logistic_regression_model_old.pkl"
joblib.dump(log_reg_classifier, model_filename)

print("Model saved as:", model_filename)

log_reg_classifier = joblib.load(model_filename)
print("Best parameters found:")
print(log_reg_classifier.named_steps['classifier'])
print("Best sampler:")
print(log_reg_classifier.named_steps['sampling'])


# -

def convert_to_labels(probas):
    labels = np.where(probas[:, 1] > 0.5, 1, -1)
    return labels


# +
y_pred_train = log_reg_classifier.predict(X_train)
y_pred_test = log_reg_classifier.predict(X_test)

sigma_values = [round(x, 1) for x in np.arange(0.1, 1.1, 0.1)]
best_sigma = None
best_f1_score = 0.0

for sigma in sigma_values:
    # Apply Gaussian smoothing
    smoothed_pred_train = gaussian_pred_smoothing(log_reg_classifier, X_train, sigma)
    smoothed_pred_test = gaussian_pred_smoothing(log_reg_classifier, X_test, sigma)
    
    # Convert smoothed probabilities to labels
    smoothed_pred_train_labels = convert_to_labels(smoothed_pred_train)
    smoothed_pred_test_labels = convert_to_labels(smoothed_pred_test)
    
    # Calculate F1 score for class -1
    f1_score_class_minus_one = f1_score(y_test, smoothed_pred_test_labels, pos_label=-1)
  
    # Update best sigma if current F1 score is higher
    if f1_score_class_minus_one > best_f1_score:
        best_f1_score = f1_score_class_minus_one
        best_sigma = sigma

print(f'Best sigma value found: {best_sigma}')

smoothed_pred_train = gaussian_pred_smoothing(log_reg_classifier, X_train, best_sigma)
smoothed_pred_test = gaussian_pred_smoothing(log_reg_classifier, X_test, best_sigma)

smoothed_pred_train_labels = convert_to_labels(smoothed_pred_train)
smoothed_pred_test_labels = convert_to_labels(smoothed_pred_test)
# -

print(smoothed_pred_train)
print(smoothed_pred_train_labels)

# +
f1_train_chirac = f1_score(y_train, y_pred_train, pos_label=1)
f1_train_mitterand = f1_score(y_train, y_pred_train, pos_label=-1)

# Calculate F1 score for each class for test set
f1_test_chirac = f1_score(y_test, y_pred_test, pos_label=1)
f1_test_mitterand = f1_score(y_test, y_pred_test, pos_label=-1)

# Calculate accuracy for training set
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate accuracy for test set
accuracy_test = accuracy_score(y_test, y_pred_test)

# Print F1 score and accuracy for each class for training set
print("Training Set:")
print("F1 Score for Chirac (label 1):", f1_train_chirac)
print("F1 Score for Mitterand (label -1):", f1_train_mitterand)
print("Accuracy:", accuracy_train)

# Print F1 score and accuracy for each class for test set
print("\nTest Set:")
print("F1 Score for Chirac (label 1):", f1_test_chirac)
print("F1 Score for Mitterand (label -1):", f1_test_mitterand)
print("Accuracy:", accuracy_test)

print('- SMOOTHED RESULTS - ')
print("Training Set:")
print("F1 Score for Chirac (label 1):", f1_score(y_train, smoothed_pred_train_labels, pos_label=1))
print("F1 Score for Mitterand (label -1):", f1_score(y_train, smoothed_pred_train_labels, pos_label=-1))
print("Accuracy:", accuracy_score(y_train, smoothed_pred_train_labels))

# Print F1 score and accuracy for each class for test set
print("\nTest Set:")
print("F1 Score for Chirac (label 1):", f1_score(y_test, smoothed_pred_test_labels, pos_label=1))
print("F1 Score for Mitterand (label -1):", f1_score(y_test, smoothed_pred_test_labels, pos_label=-1))
print("Accuracy:", accuracy_score(y_test, smoothed_pred_test_labels))
