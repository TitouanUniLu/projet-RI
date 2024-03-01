import numpy as np
import matplotlib.pyplot as plt
import os
import codecs
import re
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy.ndimage import gaussian_filter

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

fname = "./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = load_pres(fname)

#---------------------------------------------------------------------

print('OVERVIEW DATASET POUR PRESIDENTS \n')
print(f'{len(alltxts)} phrases')
print('Chirac == label 1 et Mitterand == label -1 \n')
print(f'{alltxts[0]} -> classe: {alllabs[0]}')
print(f'{alltxts[11]} -> classe: {alllabs[11]}')
print(f'Chirac: {np.sum(np.array(alllabs) == 1)} phrases - Mitterand: {np.sum(np.array(alllabs) == -1)} phrases')
print(f'on remarque que Chirac a parlé {np.round(49890/7523)} fois plus que Mitterand\n')


#fonction d'utilite
def gaussian_pred_smoothing(model, X, sigma):
        # Définition du noyau de filtre gaussien
        pred =  model.predict_proba(X)
        smoothed_pred = gaussian_filter(pred, sigma)
        return smoothed_pred

#---------------------------------------------------------------------

tfidf = TfidfVectorizer(use_idf=True, norm='l2', smooth_idf=True, lowercase=False)
X = tfidf.fit_transform(alltxts)

X_train, X_test, y_train, y_test = train_test_split(X, alllabs, test_size=0.20, random_state=12)

oversampler = SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=5) #je mets ici arbitraiement grace aux tests
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

#---------------------------------------------------------------------

num_cores = os.cpu_count()

print("Number of CPU cores available:", num_cores)
cores = 5 #hard coded temporairement pour l'entrainement

#---------------------------------------------------------------------

from imblearn.pipeline import make_pipeline, Pipeline

first_classif_v1 = make_pipeline(
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

#first_classif_v1.fit(X_train_resampled, y_train_resampled)

#---------------------------------------------------------------------

#second pipeline test
pipeline_v2 = make_pipeline(
    RandomOverSampler(sampling_strategy='minority', random_state=0),
    LogisticRegressionCV(cv=5, scoring='f1', n_jobs=cores, verbose=2, max_iter=1000)
)

param_grid_v2 = {
    'logisticregressioncv__Cs': [0.001, 0.01, 0.1, 1],
    'logisticregressioncv__penalty': ['l1', 'l2'],
    'logisticregressioncv__solver': ['liblinear', 'saga'],
    'logisticregressioncv__fit_intercept': [True],
    'logisticregressioncv__max_iter': [1000, 2000],
    'logisticregressioncv__class_weight': ['balanced', {1: 1, -1: 10}]
}

#---------------------------------------------------------------------

pipeline_v3 = Pipeline([
    ('sampling', 'passthrough'),  
    ('classifier', LogisticRegressionCV(cv=5, scoring='f1', n_jobs=cores, verbose=2, max_iter=1000))
])

    # Define the parameter grid for grid search
param_grid_v3 = {
        'sampling': [RandomOverSampler(sampling_strategy='minority', random_state=0), 
                    SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=5)],  
        'classifier__Cs': [0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__fit_intercept': [True],
        'classifier__max_iter': [2000, 3000],
        'classifier__class_weight': ['balanced', {1: 1, -1: 50}]
    }

#---------------------------------------------------------------------

# Create GridSearchCV object with scoring='f1'
model_filename = "best_logistic_regression_model.pkl"
train = False
if train:
    start_time = time.time()
    grid_search = GridSearchCV(estimator=pipeline_v3, param_grid=param_grid_v3, cv=5, scoring='f1', n_jobs=cores, verbose=2)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    print("Grid search fait en {:.2f} seconds.".format(elapsed_time))

    log_reg_classifier = grid_search.best_estimator_

    joblib.dump(log_reg_classifier, 'best_model.pkl')

    print("Model saved")

#---------------------------------------------------------------------
#loading model
log_reg_classifier = joblib.load('best_model.pkl')
print("Best parameters found:")
print(log_reg_classifier.named_steps['classifier'])
print("Best sampler:")
print(log_reg_classifier.named_steps['sampling'])

#---------------------------------------------------------------------

def convert_to_labels(probas):
    labels = np.where(probas[:, 1] > 0.5, 1, -1)
    return labels

#---------------------------------------------------------------------

y_pred_train = log_reg_classifier.predict(X_train)
y_pred_test = log_reg_classifier.predict(X_test)

#---------------------------------------------------------------------
#On cherche la meilleure valeur de sigma pour le gaussian smoothing

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

#---------------------------------------------------------------------

print(smoothed_pred_train)
print(smoothed_pred_train_labels)

#---------------------------------------------------------------------
#les f1 scores
f1_train_chirac = f1_score(y_train, y_pred_train, pos_label=1)
f1_train_mitterand = f1_score(y_train, y_pred_train, pos_label=-1)

f1_test_chirac = f1_score(y_test, y_pred_test, pos_label=1)
f1_test_mitterand = f1_score(y_test, y_pred_test, pos_label=-1)

#---------------------------------------------------------------------
#accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

#display resultats
print("F1 Score for Chirac (label 1):", f1_train_chirac)
print("F1 Score for Mitterand (label -1):", f1_train_mitterand)
print("Accuracy:", accuracy_train)

print("\nTest Set:")
print("F1 Score for Chirac (label 1):", f1_test_chirac)
print("F1 Score for Mitterand (label -1):", f1_test_mitterand)
print("Accuracy:", accuracy_test)

#---------------------------------------------------------------------

print('- SMOOTHED RESULTS - ')
print("Training Set:")
print("F1 Score for Chirac (label 1):", f1_score(y_train, smoothed_pred_train_labels, pos_label=1))
print("F1 Score for Mitterand (label -1):", f1_score(y_train, smoothed_pred_train_labels, pos_label=-1))
print("Accuracy:", accuracy_score(y_train, smoothed_pred_train_labels))

print("\nTest Set:")
print("F1 Score for Chirac (label 1):", f1_score(y_test, smoothed_pred_test_labels, pos_label=1))
print("F1 Score for Mitterand (label -1):", f1_score(y_test, smoothed_pred_test_labels, pos_label=-1))
print("Accuracy:", accuracy_score(y_test, smoothed_pred_test_labels))

#---------------------------------------------------------------------
#prediction sur le fichier test pour serveur d'eval