import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import codecs
import re
import os.path
from sklearn.model_selection import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(alltxts, alllabs, test_size=0.4, random_state=0)

president_model = joblib.load('best_model.pkl')
best_vectorizer = joblib.load('best_vectorizer.pkl')

from scipy.ndimage import gaussian_filter

def gaussian_pred_smoothing(model, X, sigma):
        # Définition du noyau de filtre gaussien
        pred =  model.predict_proba(X)
        smoothed_pred = gaussian_filter(pred, sigma)
        return smoothed_pred

pres_test = "./datasets/AFDpresidentutf8/corpus.tache1.test.utf8.txt"
# Open the file and count the number of lines/messages
with open(pres_test, "r", encoding="utf-8") as file:
    num_predictions = sum(1 for line in file)

print("Number of predictions to make:", num_predictions)

# Open the file and make predictions
with open(pres_test, "r", encoding="utf-8") as file:
    # Initialize a list to store all text data
    alltxts = []
    for line in file:
        # Strip leading/trailing whitespace and append to the list
        alltxts.append(line.strip())

print('extrait:')
print(alltxts[0])
# Transform the text data using the best vectorizer
data = best_vectorizer.transform(X_test)

probabilities = gaussian_pred_smoothing(president_model, X_test, 1)
print(probabilities)


threshold = 0.5
y_pred_binary = (probabilities > threshold).astype(int)
y_pred = np.where(probabilities[:, 1] > threshold, 1, -1)
print(y_pred)

f1_class_minus1 = f1_score(y_test, y_pred, pos_label=-1)

print("F1 Score for class -1:", f1_class_minus1)