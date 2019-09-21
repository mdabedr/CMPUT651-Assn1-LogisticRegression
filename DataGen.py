# This code is for the data generation process
import os
import numpy as np
import pickle
import pandas as pd
import time

# Paths for directories
train_pos_dir = 'G:\\Poralekha\\UofA\Fall 2019\\CMPUT 651- DL in NLP\\Assignments\\Assign 1\\imdb-movie-reviews-dataset\\aclImdb\\train\\pos\\'
train_neg_dir = 'G:\\Poralekha\\UofA\Fall 2019\\CMPUT 651- DL in NLP\\Assignments\\Assign 1\\imdb-movie-reviews-dataset\\aclImdb\\train\\neg\\'

test_pos_dir = 'G:\\Poralekha\\UofA\Fall 2019\\CMPUT 651- DL in NLP\\Assignments\\Assign 1\\imdb-movie-reviews-dataset\\aclImdb\\test\\pos\\'
test_neg_dir = 'G:\\Poralekha\\UofA\Fall 2019\\CMPUT 651- DL in NLP\\Assignments\\Assign 1\\imdb-movie-reviews-dataset\\aclImdb\\test\\neg\\'
# Variables to load data
train_pos = []
train_neg = []
test_pos = []
test_neg = []


# Load text from all of the text files
def loadText(dir, variable):
    for f in os.listdir(dir):
        print (f)
        if os.path.isfile(dir + f):
            variable.append(dir + f)


# Load the text document names into variables
loadText(train_pos_dir, train_pos)
loadText(train_neg_dir, train_neg)
loadText(test_pos_dir, test_pos)
loadText(test_neg_dir, test_neg)


# Reading the text data
def readtext(X):
    for idx, fname in enumerate(X):
        with open(fname) as f:
            print(fname)
            text = f.readline().strip().lower()
        X[idx] = text
    return None


# Caling readtext function
readtext(train_pos)
readtext(train_neg)
readtext(test_pos)
readtext(test_neg)

# generate ytest and raw Xtest
test = test_pos + test_neg
y_test = [1] * len(test_pos) + [0] * len(test_neg)

# shuffle training data
np.random.shuffle(train_pos)
np.random.shuffle(train_neg)

# Separate validation pos and validation neg
val_pos = train_pos[0:2500]
val_neg = train_neg[0:2500]

# Generate raw training set
X1 = train_pos[2500:12500]
X2 = train_neg[2500:12500]
X = list()
X.extend(X1)
X.extend(X2)
X_train = X
y_train = [1] * len(X1) + [0] * len(X2)

# Generate raw validation set
val = val_pos + val_neg
y_val = [1] * len(val_pos) + [0] * len(val_neg)

# Reshuffle raw training set
np.random.seed(320)
np.random.shuffle(X_train)
np.random.seed(320)
np.random.shuffle(y_train)

# Generate vocabulary
vocab = dict()

for i in X_train:
    for w in i.split():
        if w in vocab:
            vocab[w] += 1
        else:
            vocab[w] = 1

vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
vocab_to_consider = vocab[0:2000]

# Make the word2id and id2word dictionary
word2id = dict()
id2word = dict()
count = 0
for i in vocab_to_consider:
    word2id[i[0]] = count
    print(word2id[i[0]], count)
    id2word[count] = i[0]
    print(count, id2word[count])

    print(i[0], count)
    count += 1


# Function to turn raw data into dataset
def makeDataSet(vocab_to_consider, X, word2id):
    names = []

    for i in vocab_to_consider:
        names.append(i[0])
    ajaira = [0] * len(names)
    df = pd.DataFrame(columns=names)
    for i in X:
        i = i.split()
        for j in i:
            print(j)
            if j in word2id:
                print('yes ' + j + ' is in vocab_to_consider')
                ajaira[word2id[j]] = 1

        print (ajaira)
        temp = pd.DataFrame([ajaira], columns=names)
        df = df.append(temp)
        print(len(df))
        # time.sleep(1)
        ajaira = [0] * len(names)
    return df


# Generate training, test and validation sets by calling the makeData function
X_train = makeDataSet(vocab_to_consider, X_train, word2id)
X_val = makeDataSet(vocab_to_consider, val, word2id)
X_test = makeDataSet(vocab_to_consider, test, word2id)

# Dump the training, validation and test sets
pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), open("Important2.pk", "wb"))
