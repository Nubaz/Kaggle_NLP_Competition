# I - Pregatirea datelor de antrenare si de testare
# - Importul librariilor de baza si incarcarea datelor de antrenare/testare
import os
import numpy as np
import pandas as pd

data_path = '../Proiect/data/'
orig_train_data = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
orig_test_data = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

# - Preprocesarea datelor de antrenare
# -- Transformarea etichetelor/limbilor in id-uri numerice
unique_labels = orig_train_data['label'].unique()
unique_langs = orig_train_data['language'].unique()

label2id = {}
id2label = {}
for idx, label in enumerate(unique_labels):
    label2id[label] = idx
    id2label[idx] = unique_labels[idx]

lang2id = {}
id2lang = {}
for idx, lang in enumerate(unique_langs):
    lang2id[lang] = idx
    id2lang[idx] = unique_langs[idx]

labels = []
for i in orig_train_data['label']:
    labels.append(label2id[i])

langs = []
for i in orig_train_data['language']:
    langs.append(lang2id[i])

# -- Functia de preprocesare a unui text dat
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def preproc(text, language=None):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    text = text.translate(str.maketrans('', '', "-'"))
    text = word_tokenize(text.lower())

    if language != None:
        stemmer = SnowballStemmer(language)
        stop_words = set(stopwords.words(language))
        filtered_text = []
        
        for word in text:
            if word not in stop_words:
                filtered_text.append(stemmer.stem(word))
        return filtered_text
    else:
        return text

# -- Delimitarea range-urilor de indecsi pt. cele 5 limbi
interv = np.linspace(0, 41570, 6)
limits = []
for i in range(6):
    limits.append(int(interv[i]))

# -- Construirea noului dataframe de train, cu textele preprocesate si etichetele transforamte in id-uri
def preproc_data(orig_train_data):
    idx = 0
    data = []
    for text in orig_train_data['text']:
        if idx < limits[1]:
            data.append(preproc(text, 'danish'))
            idx += 1
        elif idx < limits[2]:
            data.append(preproc(text, 'german'))
            idx += 1
        elif idx < limits[3]:
            data.append(preproc(text, 'spanish'))
            idx += 1
        elif idx < limits[4]:
            data.append(preproc(text, 'italian'))
            idx += 1
        else:
            data.append(preproc(text, 'dutch'))
            idx += 1

    preproc_train_data = pd.DataFrame({
        "language": langs,
        "text": data,
        "label": labels,
    })

    return preproc_train_data

preproc_train_data = preproc_data(orig_train_data)

# - Preprocesarea datelor de testare
from langdetect import detect

preproc_test_data = []
for text in orig_test_data['text']:
    if text != orig_test_data["text"].iloc[7582]:
        lang_detect = detect(text)
        if lang_detect == "da":
            preproc_test_data.append(preproc(text, "danish"))
        elif lang_detect == "de":
            preproc_test_data.append(preproc(text, "german"))
        elif lang_detect == "es":
            preproc_test_data.append(preproc(text, "spanish"))
        elif lang_detect == "it":
            preproc_test_data.append(preproc(text, "italian"))
        elif lang_detect == "nl":
            preproc_test_data.append(preproc(text, "dutch"))
        else:
            preproc_test_data.append(preproc(text))
    else:
        preproc_test_data.append("empty")

# II - Definirea modelului Bag-of-Words
from collections import Counter

def count_most_common(how_many, preproc_texts):
    ctr = Counter()
    for word in preproc_texts:
        ctr.update(word)
    keywords = []
    for word, freq in ctr.most_common(how_many):
        if word.strip():
            keywords.append(word)
    return keywords

def build_id_word_dicts(keywords):
    word2id = {}
    id2word = {}
    for idx, word in enumerate(keywords):
        word2id[word] = idx
        id2word[idx] = word
    return word2id, id2word

def featurize(preproc_text, id2word):
    ctr = Counter(preproc_text)
    features = np.zeros(len(id2word))
    for idx in range(0, len(features)):
        word = id2word[idx]
        features[idx] = ctr[word]
    return features

def featurize_multi(texts, id2word):
    all_features = []

    for text in texts:
        all_features.append(featurize(text, id2word))
    return np.array(all_features)

def normalizer(X):
    rows = np.shape(X)[0]
    cols = np.shape(X)[1]

    X_norm = np.zeros((rows, cols))
    X_norm_arr = np.sqrt(np.sum(X**2, axis=1))
    for i in range(rows):
        for j in range(cols):
            if X_norm_arr[i] != 0:
                X_norm[i][j] = X[i][j] / X_norm_arr[i]
            else:
                X_norm[i][j] = X[i][j]
    return X_norm

# III - Metoda k-fold de testare
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def k_fold_testing(folds, how_many, model):
    acc_score = []
    conf_matrix = []

    kf = KFold(folds, shuffle=True)
    for train_idx, valid_idx in kf.split(preproc_train_data):
        train_data = preproc_train_data.loc[train_idx]
        valid_data = preproc_train_data.loc[valid_idx]
        
        keywords = count_most_common(how_many, train_data["text"].to_list())
        word2id, id2word = build_id_word_dicts(keywords)

        X_train = featurize_multi(train_data["text"].to_list(), id2word)
        X_valid = featurize_multi(valid_data["text"].to_list(), id2word)        
        
        X_train = normalizer(X_train)
        X_valid = normalizer(X_valid)

        model.fit(X_train, train_data["label"])
        pred_values = model.predict(X_valid)

        acc = accuracy_score(valid_data["label"], pred_values)
        acc_score.append(acc)
        conf_m = confusion_matrix(valid_data["label"], pred_values)
        conf_matrix.append(conf_m)

    return acc_score, conf_matrix

# IV - Testarea algoritmului si aplicarea pe toate datele de antrenare
def predict_values(model, how_many):
    keywords = count_most_common(how_many, preproc_train_data["text"].to_list())
    word2id, id2word = build_id_word_dicts(keywords)

    X_train = featurize_multi(preproc_train_data["text"].to_list(), id2word)
    X_test = featurize_multi(preproc_test_data, id2word)

    X_train_norm = normalizer(X_train)
    X_test_norm = normalizer(X_test)

    model.fit(X_train_norm, preproc_train_data["label"].to_list())
    pred_values = model.predict(X_test_norm)

    return pred_values

from sklearn.neighbors import KNeighborsClassifier as kNN
model = kNN(7)

acc_score, conf_matrix = k_fold_testing(5, 5000, model)
print(acc_score)
print(sum(acc_score) / 5)
print(np.round(sum(conf_matrix) / 5))

pred_values = predict_values(model, 5000)

# V - Exportarea etichetelor prezise
pred_values_lit = []
for idx in range(len(pred_values)):
    pred_values_lit.append(id2label[pred_values[idx]])

id = np.arange(1, 13861)
np.savetxt("../Proiect/data/test_labels.csv", np.stack((id, pred_values_lit)).T, fmt="%s", delimiter=",", header="id,label", comments= '')