# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import matplotlib.pyplot as plt

import codecs
import re
import os.path


# -

# # Données reconnaissance du locuteur (Chirac/Mitterrand)

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

print(f'{len(alltxts)} phrases')
print('Chirac == label 1 et Mitterand == label -1')
print(f'{alltxts[0]} -> classe: {alllabs[0]}')
print(f'{alltxts[11]} -> classe: {alllabs[11]}')
print(f'Chirac: {np.sum(np.array(alllabs) == 1)} phrases - Mitterand: {np.sum(np.array(alllabs) == -1)} phrases')
print(f'on remarque que Chirac a parlé {np.round(49890/7523)} fois plus que Mitterand')


# # Données classification de sentiments (films)

def load_movies(path2data): # 1 classe par répertoire
    alltxts = [] # init vide
    labs = []
    cpt = 0
    for cl in os.listdir(path2data): # parcours des fichiers d'un répertoire
        for f in os.listdir(path2data+cl):
            txt = open(path2data+cl+'/'+f).read()
            alltxts.append(txt)
            labs.append(cpt)
        cpt+=1 # chg répertoire = cht classe
        
    return alltxts,labs



# +
path = "./datasets/movies/movies1000/"

alltxts_mov,alllabs_mov = load_movies(path)
# -

print(len(alltxts_mov),len(alllabs_mov))
print(alltxts_mov[0])
print(alllabs_mov[0])
print(alltxts_mov[-1])
print(alllabs_mov[-1])

# # A) Transformation paramétrique du texte (pre-traitements)
#
# Vous devez tester, par exemple, les cas suivants:
# - transformation en minuscule ou pas
# - suppression de la ponctuation
# - transformation des mots entièrement en majuscule en marqueurs spécifiques
# - suppression des chiffres ou pas
# - conservation d'une partie du texte seulement (seulement la première ligne = titre, seulement la dernière ligne = résumé, ...)
# - stemming
# - ...
#
#
# Vérifier systématiquement sur un exemple ou deux le bon fonctionnement des méthodes sur deux documents (au moins un de chaque classe).

# +
import string 
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
nltk.download('punkt')

punc = string.punctuation 
punc += '\n\r\t'
stemmer = FrenchStemmer()

for i in range(0,len(alltxts)):
    #alltxts[i] = re.sub('[0-9]+', '', alltxts[i]) #a garder ou enlever
    alltxts[i] = alltxts[i].lower() 
    alltxts[i] = alltxts[i].translate(str.maketrans(punc, ' ' * len(punc))) #ponctuation
    alltxts[i] = unicodedata.normalize('NFD', alltxts[i]).encode('ascii', 'ignore').decode("utf-8") #normalize en unicode, enleve non ascii et reconvertir en utf 8
    words = word_tokenize(alltxts[i], language='french')
    stemmed = [stemmer.stem(word) for word in words]
    alltxts[i] = ' '.join(stemmed)

print('Processed:')
print(f'\nclass: {alllabs[0]} , texte: {alltxts[0]}')
print(f'\nclass: {alllabs[11]} , texte: {alltxts[11]}')
# -

# # B) Extraction du vocabulaire (BoW)
#
# - **Exploration préliminaire des jeux de données**
#     - Quelle est la taille d'origine du vocabulaire?
#     - Que reste-t-il si on ne garde que les 100 mots les plus fréquents? [word cloud]
#     - Quels sont les 100 mots dont la fréquence documentaire est la plus grande? [word cloud]
#     - Quels sont les 100 mots les plus discriminants au sens de odds ratio? [word cloud]
#     - Quelle est la distribution d'apparition des mots (Zipf)
#     - Quels sont les 100 bigrammes/trigrammes les plus fréquents?
#
# - **Variantes de BoW**
#     - TF-IDF
#     - Réduire la taille du vocabulaire (min_df, max_df, max_features)
#     - BoW binaire
#     - Bi-grams, tri-grams
#     - **Quelles performances attendre ? Quels sont les avantages et les inconvénients des ces variantes?**

# +
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

''' Exploration préliminaire des jeux de données '''
nltk.download('stopwords')

all_words = ' '.join(alltxts).split()

# frequence de chaque mot avec counter
word_freq = Counter(all_words)

# Quelle est la taille d'origine du vocabulaire?
original_vocab_size = len(word_freq)
print(f"Taille d'origine du vocabulaire: {original_vocab_size}")
'''
# +
# - Que reste-t-il si on ne garde que les 100 mots les plus fréquents? [word cloud]
#removed stopwords
vectorizer = CountVectorizer(stop_words=stopwords.words("french"))
X = vectorizer.fit_transform(alltxts)
frequent_words = pd.Series(
    np.array(X.sum(axis=0))[0], index=sorted(vectorizer.vocabulary_)
)

wordcloud = WordCloud(background_color="white", max_words=100, width=2000, height=1000)
wordcloud.generate_from_frequencies(frequent_words)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

print("20 most frequent words")
print(frequent_words.sort_values(ascending=False)[:20])
print(len(frequent_words))
print(sum(frequent_words))

# +
#kept stopwords
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(alltxts)
frequent_words = pd.Series(
    np.array(X.sum(axis=0))[0], index=sorted(vectorizer.vocabulary_)
)

wordcloud = WordCloud(background_color="white", max_words=100, width=2000, height=1000)
wordcloud.generate_from_frequencies(frequent_words)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

print("20 most frequent words")
print(frequent_words.sort_values(ascending=False)[:20])
print(len(frequent_words))
print(sum(frequent_words))

# +
#Quelle est la distribution d'apparition des mots (Zipf)
word_counts = np.array(X.sum(axis=0))[0]
sorted_indices = np.argsort(word_counts)[::-1] 

sorted_word_counts = word_counts[sorted_indices]
plt.figure(figsize=(12, 6))
plt.plot(sorted_word_counts, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.title('Distribution de Zipf des mots')
plt.xlabel('Rang des mots (log)')
plt.ylabel('Fréquence des mots (log)')
plt.show()


# +
#- Quels sont les 100 mots dont la fréquence documentaire est la plus grande? [word cloud]
#removed stopwords
vectorizer = TfidfVectorizer(stop_words=stopwords.words("french"))
X = vectorizer.fit_transform(alltxts)
frequent_words = pd.Series(
    np.array(X.sum(axis=0))[0], index=sorted(vectorizer.vocabulary_)
)

wordcloud = WordCloud(background_color="white", max_words=100, width=2000, height=1000)
wordcloud.generate_from_frequencies(frequent_words)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

print("20 most frequent words")
print(frequent_words.sort_values(ascending=False)[:20])
print(len(frequent_words))
print(sum(frequent_words))

# +
#kept stopwords
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(alltxts)
frequent_words = pd.Series(
    np.array(X.sum(axis=0))[0], index=sorted(vectorizer.vocabulary_)
)

wordcloud = WordCloud(background_color="white", max_words=100, width=2000, height=1000)
wordcloud.generate_from_frequencies(frequent_words)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

print("20 most frequent words")
print(frequent_words.sort_values(ascending=False)[:20])
print(len(frequent_words))
print(sum(frequent_words))

# +
#Quels sont les 100 bigrammes/trigrammes les plus fréquents?
#removed stopwords
vectorizer = CountVectorizer(stop_words=stopwords.words("french"), ngram_range=(2,3)) 
X = vectorizer.fit_transform(alltxts)
frequent_words = pd.Series(
    np.array(X.sum(axis=0))[0], index=sorted(vectorizer.vocabulary_)
)

wordcloud = WordCloud(background_color="white", max_words=100, width=2000, height=1000)
wordcloud.generate_from_frequencies(frequent_words)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

print("20 most frequent bigrams and trigrams")
print(frequent_words.sort_values(ascending=False)[:20])
print(len(frequent_words))
print(sum(frequent_words))


# +
#kept stopwords
vectorizer = CountVectorizer(ngram_range=(2,3)) 
X = vectorizer.fit_transform(alltxts)
frequent_words = pd.Series(
    np.array(X.sum(axis=0))[0], index=sorted(vectorizer.vocabulary_)
)

wordcloud = WordCloud(background_color="white", max_words=100, width=2000, height=1000)
wordcloud.generate_from_frequencies(frequent_words)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

print("20 most frequent bigrams and trigrams")
print(frequent_words.sort_values(ascending=False)[:20])
print(len(frequent_words))
print(sum(frequent_words))

# +
#- Quels sont les 100 mots les plus discriminants au sens de odds ratio? [word cloud]


# -
'''
# # C) Modèles de Machine Learning

# ## 1) Métriques d'évaluation 
#
# Il faudra utiliser des métriques d'évaluation pertinentes suivant la tâche et l'équilibrage des données : 
# - Accuracy
# - Courbe ROC, AUC, F1-score

# +
''' On teste trois modèles: Naive Bayes, SVM, Logistic Regression'''

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


X = CountVectorizer(stop_words=stopwords.words("french")).fit_transform(alltxts)
X_train, X_test, y_train, y_test = train_test_split(X, alllabs, test_size=0.4, random_state=0)


def tests_models(X_train, X_test, y_train, y_test, class_weights=None, smoothing=None):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)

    logistic_reg = LogisticRegression(class_weight=class_weights, n_jobs=-1) #n_jobs = -1 pour utiliser tous les processeurs
    logistic_reg.fit(X_train, y_train)

    svm = LinearSVC(class_weight=class_weights)
    svm.fit(X_train, y_train)

    if smoothing is None:
        pred_nb = naive_bayes.predict(X_test)
        pred_lr = logistic_reg.predict(X_test)
        pred_svm = svm.predict(X_test)
    #gaussian smoothing, on passe la fonction dans l'attribut smoothing de tests_models
    else:
        try:
            smoothing_size = 5 #a modifier pour voir l'impact
            pred_nb = smoothing(naive_bayes.predict_proba(X_test)[:, 1], smoothing_size)
            pred_lr = smoothing(logistic_reg.predict_proba(X_test)[:, 1], smoothing_size)
            pred_svm = smoothing(svm.decision_function(X_test), smoothing_size)
            
            # threshold sinon on a un mix de valeurs binaires et continues
            pred_nb = (pred_nb > 0.5).astype(int)
            pred_lr = (pred_lr > 0.5).astype(int)
            pred_svm = (pred_svm > 0.5).astype(int)
        except Exception as e:
            print("Error with the gaussian smoothing:", e)


    #fonction de scikit pour print les metriques interresantes
    print(f"NB accuracy: \n {classification_report(y_test, pred_nb, zero_division=1)}")
    print(f"LR accuracy: \n {classification_report(y_test, pred_lr, zero_division=1)}")
    print(f"SVM accuracy: \n {classification_report(y_test, pred_svm, zero_division=1)}")

    #on recupere les faux positifs et vrai positifs pour chaque modele
    #pour ca on utilise la proba d'apartenir a une classe plutot que la classe predite 
    fp_naive_bayes, vp_naive_bayes, _ = roc_curve(y_test, naive_bayes.predict_proba(X_test)[:, 1]) #on recupere la proba de la classe "positive"
    fp_logistic_reg, vp_logistic_reg, _ = roc_curve(y_test, logistic_reg.predict_proba(X_test)[:, 1])
    fp_svm, vp_svm, _ = roc_curve(y_test, svm.decision_function(X_test))
    #le _ ici permet d'ignorer la veleur du threshold comme on l'utilise pas ici

    roc_auc_nb = auc(fp_naive_bayes, vp_naive_bayes)
    roc_auc_lr = auc(fp_logistic_reg, vp_logistic_reg)
    roc_auc_svm = auc(fp_svm, vp_svm)

    plt.figure(figsize=(10, 8))
    plt.plot(fp_naive_bayes, vp_naive_bayes, color='blue', lw=2, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})') #le :.2f permet d'avoir la valeur au centieme pres
    plt.plot(fp_logistic_reg, vp_logistic_reg, color='green', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})') 
    plt.plot(fp_svm, vp_svm, color='red', lw=2, label=f'SVM (AUC = {roc_auc_svm:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--') #random classifier
    plt.xlim([0.0, 1.0]) #limite des valeurs pour les axes
    plt.ylim([0.0, 1.01]) #je mets un peu plus que 1 ici sinon la visibilité est pas parfaite
    plt.xlabel('Faux Positifs')
    plt.ylabel('Vrai Positifs')
    plt.title('ROC')
    plt.legend(loc="lower right") #legende en bas a droite
    plt.show()

#tests_models(X_train, X_test, y_train, y_test)

# -

# <u>Naive-Bayes:</u>
#
# Resultat Accuracy et f1-score:
# Dans la partie 1 on avait vu que la classe 1 (Chirac) était beaucoup plus présente que la classe -1. Pour un NB, c'est donc logique que la precision pour la premiere classe soit de 91% mais que de 58% pour la classe -1. On retrouve la meme chose pour le f1-score. Cela est donc clairmeent du au fait que les classes ne sont pas reparties equitablement. Il faudrait faire du oversampling sur la classe Miterrand pour avoir une meilleure évaluation de NB.
#
# Resultat ROC:
# On voit que la courbe ROC de NB se raproche du coté en haut à gauche, qui est donc le scénario ideal ou le rate de vrai positifs est de 1 (donc 100% de performance). 
#
# Resultat AUC:
# NB obtient un score de 0.83 pour l'aire en dessous de la courbe. Donc NB arrive plutot bien a séparer les instances de la classe 1 et -1. 
#
# Au final on voit que le fait qu'il y ait beaucoup plus d'instances de la classe 1 manipule les résultats car les performances pour cette classe sont bien meilleure que pour la classe -1.
#
# <u>Logistic Regression:</u>
#
# Resultat Accuracy et f1-score:
# Pour la regression logistique on obtient la meme precision pour la classe 1 qu'avec NB mais un accuracy plus élevé pour la classe -1. L'accuracy globale pour LR est de 90% (contre 88% pour NB). Le f1-score est très légerement supérieur pour LR que pour NB pour les deux classe. 
#
# Resultat ROC:
# Meme analyse que pour NB, mais avec une meilleure performance.
#
# Resultat AUC:
# Meme analyse que pour NB, mais avec une meilleure performance.
#
# <u>SVM:</u>
#
# Resultat Accuracy et f1-score:
#
# Resultat ROC:
# Meme analyse que pour NB, mais avec une moins bonne performance que NB et LR.
#
# Resultat AUC:
# Meme analyse que pour NB, mais avec une moins bonne performance que NB et LR.

# ## 2) Variantes sur les stratégies d'entraînement
#
# - **Sur-apprentissage**. Les techniques sur lesquelles nous travaillons étant sujettes au sur-apprentissage: trouver le paramètre de régularisation dans la documentation et optimiser ce paramètre au sens de la métrique qui vous semble la plus appropriée (cf question précédente).
#
#  <br>
# - **Equilibrage des données**. Un problème reconnu comme dur dans la communauté est celui de l'équilibrage des classes (*balance* en anglais). Que faire si les données sont à 80, 90 ou 99% dans une des classes?
# Le problème est dur mais fréquent; les solutions sont multiples mais on peut isoler 3 grandes familles de solution.
#
# 1. Ré-équilibrer le jeu de données: supprimer des données dans la classe majoritaire et/ou sur-échantilloner la classe minoritaire.<BR>
#    $\Rightarrow$ A vous de jouer pour cette technique
# 1. Changer la formulation de la fonction de coût pour pénaliser plus les erreurs dans la classe minoritaire:
# soit une fonction $\Delta$ mesurant les écarts entre $f(x_i)$ et $y_i$ 
# $$C = \sum_i  \alpha_i \Delta(f(x_i),y_i), \qquad \alpha_i = \left\{
# \begin{array}{ll}
# 1 & \text{si } y_i \in \text{classe majoritaire}\\
# B>1 & \text{si } y_i \in \text{classe minoritaire}\\
# \end{array} \right.$$
# <BR>
#    $\Rightarrow$ Les SVM et d'autres approches sklearn possèdent des arguments pour régler $B$ ou $1/B$... Ces arguments sont utiles mais pas toujours suffisant.
# 1. Courbe ROC et modification du biais. Une fois la fonction $\hat y = f(x)$ apprise, il est possible de la *bidouiller* a posteriori: si toutes les prédictions $\hat y$ sont dans une classe, on va introduire $b$ dans $\hat y = f(x) + b$ et le faire varier jusqu'à ce qu'un des points change de classe. On peut ensuite aller de plus en plus loin.
# Le calcul de l'ensemble des scores associés à cette approche mène directement à la courbe ROC.
#
# **Note:** certains classifieurs sont intrinsèquement plus résistante au problème d'équilibrage, c'est par exemple le cas des techniques de gradient boosting que vous verrez l'an prochain.

# +
''' Oversampling sur la classe -1 (Mitterand), donc on va dupliquer les examples de la classe minoritaire'''
from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(sampling_strategy='minority')
X_over_train, Y_over_train = over_sampler.fit_resample(X_train, y_train) #on fait du over sampling sur les echantillons train et test en bas
X_over_test, Y_over_test = over_sampler.fit_resample(X_test, y_test)
#print(f'''Nombre d\'éléments de la classe -1 après over sampling: {Y_over_train.count(-1)},
#Nombre d\'éléments de la classe 1 après over sampling: {Y_over_train.count(1)}''')
assert(Y_over_train.count(-1) == Y_over_train.count(1)) #on verifie que l'over sampling a bien été fait

#on refait des tests sur les modèles avec nos données oversamplées
#tests_models(X_over_train, X_over_test, Y_over_train, Y_over_test)
# -

# Après Oversampling, on observe une nette améliorations des modèles pour la classe -1 pour toute les métriques (precision, recall, f1 score). Pour la classe 1, on remarque aussi que les performances sont moins bonnes. Cela est logique est s'explique par la diminution du biais qui était present a cause de la domination de la classe 1 pendant les premiers tests. Avec l'oversampling, la classe minoritaire peu etre mieux détectée. 
#
# L'accuracy global pour chaque modèle a baissé mais c'ets normal car les modèles sont moins biaisé (vis a vis de la classe 1). La diminition de l'accuracy n'est pas un problème pour cette situation.
#
# La courbe ROC et l'AUC reste quasi inchangée, et cela peut s'expliquer par le fait que les modèles arrivaient deja a séparer les classes plutot bien, donc l'oversampling n'a pas changé grand chose à ce problème.

# +
''' Undersampling sur la classe 1 (Chirac), donc on va suprrimer des exemples de la classe 1 pour equilibrer les données'''
from imblearn.under_sampling import RandomUnderSampler
under_sampler = RandomUnderSampler(sampling_strategy='majority')
X_under_train, Y_under_train = under_sampler.fit_resample(X_train, y_train) 
X_under_test, Y_under_test = under_sampler.fit_resample(X_test, y_test)
#print(f'''Nombre d\'éléments de la classe -1 après under sampling: {Y_under_train.count(-1)},
#Nombre d\'éléments de la classe 1 après under sampling: {Y_under_train.count(1)}''')
assert(Y_under_train.count(-1) == Y_under_train.count(1)) #on verifie que l'under sampling a bien été fait

#memes tests qu'avant
#tests_models(X_under_train, X_under_test, Y_under_train, Y_under_test)
# -

# D'abord, on peut observer qu'avec du under sampling, on se retrouve avec beaucoup moins d'exemples: 4500 par classe pour du under sampling contre 30000 pour du over sampling. 
#
# Pourtant, on observe des resultats similaires: la precision de la classe -1 augmente beaucoup et la precision de la classe 1 diminue légerement. Pareil pour la courbe ROC et l'AUC. Cela peut s'expliquer par le fait que le biais envers la classe 1 soit reduit et que la classe -1 peut donc etre mieux apprise.
#
# Concernant l'accuracy, les résultats sont presque identiques: si on regarde au centième près, c'est effectivment l'oversampling qui obtient de meilleurs resultats (mais cette difference est negligeable). Pareil pour l'AUC.

''' Penalized Cost model sur la classe minoritaire -1'''
from sklearn.utils.class_weight import compute_class_weight
#ici class_weights permet d'assigner des poids élevés aux exemples de la classe minoritaire pour 
#que des erreurs sur cette classe soit plus pénalisée (la classe 1 a des poids plus faibles)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = class_weights = dict(zip(np.unique(y_train), class_weights))
#tests_models(X_train, X_test, y_train, y_test, class_weights)

# En penalisant les erreurs sur la classe minoritaires, on retrouve quand meme des resultats par tres performants pour la classe -1. Si on compare avec un le premier test (partie C.1) on observe que les resultats pour la classe -1 sont moins bon pour la precision et l'accuracy, a peu pres pareil pour le f1-score et très legerment meilleur pour l'AUC.

# ## 3) Post-processing sur les données Président
#
# Pour la tâche de reconnaissance de locuteur, des phrases successives sont souvent associés à un même locuteur. Voir par exemples les 100 premiers labels de la base d'apprentissage. 

# +
fname = "./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = load_pres(fname)

#plt.figure()
#plt.plot(list(range(len(alllabs[0:100]))),alllabs[0:100])
# -

# **Une méthode de post-traitement pour améliorer les résultats consistent à lisser les résultats de la prédictions d'une phrases par les prédictions voisines, en utilisant par exemple une convolution par une filtre Gaussien. Compléter la fonction ci-dessous et tester l'impact de ce lissage sur les performances.**

# +
from scipy.ndimage import gaussian_filter

def gaussian_pred_smoothing(model, X, sigma):
        # Définition du noyau de filtre gaussien
        pred =  model.predict_proba(X)
        smoothed_pred = gaussian_filter(pred, sigma)
        return smoothed_pred



# -

#
# ## 4) Estimer les performances de généralisation d'une méthodes
# **Ce sera l'enjeu principal du projet : vous disposez d'un ensemble de données, et vous évaluerez les performances sur un ensemble de test auquel vous n'avez pas accès. Il faut donc être capable d'estimer les performances de généralisation du modèles à partir des données d'entraînement.**

#
# Avant de lancer de grandes expériences, il faut se construire une base de travail solide en étudiant les questions suivantes:
#
# - Combien de temps ça prend d'apprendre un classifieur NB/SVM/RegLog sur ces données en fonction de la taille du vocabulaire?
# - La validation croisée est-elle nécessaire? Est ce qu'on obtient les mêmes résultats avec un simple *split*?
# - La validation croisée est-elle stable? A partir de combien de fold (travailler avec différentes graines aléatoires et faire des statistiques basiques)?

# +
''' fonctions d'utilités'''

def plot_metrics(k_values, avg_train_time, avg_test_accuracy):
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_train_time, marker='o', linestyle='-')
    plt.xlabel('k')
    plt.ylabel('Temps d\'entrainement moyen')
    plt.title('Temps d\'entrainement pour chaque k')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_test_accuracy, marker='o', linestyle='-')
    plt.xlabel('k')
    plt.ylabel('Performance moyenne')
    plt.title('Performance moyenne pour chaque k')
    plt.grid(True)
    plt.show()


# +

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
import joblib

X_train, X_test, y_train, y_test = train_test_split(alltxts, alllabs, test_size=0.4, random_state=0)
import csv

# Assuming X_train and y_train are your training features and labels, respectively
# Save the training data to a CSV file
'''with open('training_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(['Features', 'Label'])
    # Write each feature-label pair
    for features, label in zip(X_train, y_train):
        writer.writerow([features, label])'''

# Define the pipeline to search for the best model and vectorizer
logistic_regression_pipeline = ImbPipeline([
    ('vectorizer', CountVectorizer()),  # This will be tested with both CountVectorizer and TfidfVectorizer
    ('scaler', StandardScaler(with_mean=False)),  # Add StandardScaler for scaling
    ('sampler', RandomOverSampler(sampling_strategy='minority')),  # Add RandomOverSampler for oversampling
    ('classifier', LogisticRegression())
])

# Define the parameters to optimize for Logistic Regression and the Vectorizer
param_grid = {
    'vectorizer': [CountVectorizer(), TfidfVectorizer()],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],  # Test different ngram ranges
    'vectorizer__max_features': [None],  # Test different maximum features          1000, 5000, 
    'classifier__C': [0.1, 1, 10],  # Inverse of regularization strength
    'classifier__penalty': ['l2'],  # Regularization norm
    'classifier__solver': ['liblinear', 'saga'],  # Algorithm to use in optimization
    'classifier__max_iter': [2000, 3000]  # Maximum number of iterations  100, 1000, 
}

# Perform a grid search to find the best model and vectorizer
grid_search = GridSearchCV(logistic_regression_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=10)
print('Fitting data...')
grid_search.fit(X_train, y_train)

# Get the best model and best vectorizer
best_model = grid_search.best_estimator_
best_vectorizer = grid_search.best_estimator_.named_steps['vectorizer']

print("Best Model:", best_model.__class__.__name__)
print("Best Vectorizer:", best_vectorizer.__class__.__name__)

# Print the best parameters found
print("Best parameters:", grid_search.best_params_)

# Save the best model and best vectorizer
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(best_vectorizer, 'best_vectorizer.pkl')
# -

president_model = joblib.load('best_model.pkl')
best_vectorizer = joblib.load('best_vectorizer.pkl')
pred = president_model.predict(X_test)
print(classification_report(y_test, pred, zero_division=1))
print("Classes:", president_model.classes_)

sigma = 1
probabilities_test = gaussian_pred_smoothing(president_model, X_test, sigma)

# Convert probabilities to binary predictions using a threshold of 0.5
threshold = 0.5
y_pred_test = np.where(probabilities_test[:, 1] > threshold, 1, -1)

# Print classification reportpython 
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred_test, zero_division=1))


# On observe que le temps d'entrainement diminue légerement quand la taille du vocabulaire augmente. Cela s'epxlique par le fait que scikit-learn et CountVectorizer sont optimisés pour gérer des des matrices de features de grande taille.

# +
'''
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

# Transform the text data using the best vectorizer
data = best_vectorizer.transform(alltxts)

print('converting text...')
# Convert the sparse matrix into a list of strings
text_data = best_vectorizer.inverse_transform(data)
text_data = [" ".join(words) for words in text_data]
print('text converted')

# Make predictions on the transformed data
#probabilities = president_model.predict_proba(text_data)
probabilities = gaussian_pred_smoothing(president_model, text_data, 1)

print(f"Made {len(probabilities)} predictions")
print("Classes:", president_model.classes_)
# Save predicted probabilities to a file
with open("predictions.txt", "w") as file:
    for prob in probabilities:
        # Write the probability of the negative class (-1)
        file.write(f"{prob[0]}\n")

print(probabilities[0:10])


# +
import time

# on peut aussi augmenter la taille du vocab en rajoutant des ngrams
print(f"taille du vocabulaire: {len(best_vectorizer.get_feature_names_out())}") 

# on va utiliser k = 8 ici
vect_ngram = CountVectorizer(ngram_range=(1, 3))
vect_ngram.fit(X_train)
print(f"taille du vocabulaire avec ngrams: {len(vect_ngram.get_feature_names_out())}")

taille_vocab = []
temps_entrainement = []
# test sur le vocab sans ngram pour le moment 
# pour tester avec les ngrams, il faut juste changer "vect" en "vect_ngram" juste ici 
# et changer le step et init de la loop
for feature in range(10000, len(best_vectorizer.get_feature_names_out()), 1000):
    vectorizer = CountVectorizer(max_features=feature)

    start_time = time.time() #init le temps
    vectorizer.fit(X_train) #train
    end_time = time.time()
    elapsed_time = end_time - start_time
    taille_vocab.append(feature)
    temps_entrainement.append(elapsed_time)
    #print(feature, elapsed_time)
if False:
    plt.plot(taille_vocab, temps_entrainement, marker='o')
    plt.xlabel('Taille du vocabulaire')
    plt.ylabel('Temps d\'apprentissage en sec')
    plt.title('Temps d\'apprentissage en fonction de la taille du vocabulaire')
    plt.xlim(min(taille_vocab), max(taille_vocab)+0.1)  
    plt.ylim(min(temps_entrainement)-0.5, max(temps_entrainement))
    plt.grid(True)
    plt.show()
'''