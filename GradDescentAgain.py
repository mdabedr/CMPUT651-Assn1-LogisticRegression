# coding=utf-8

# Get everything from dump to initialize

# Load everything
# df, X_train, y_train, val_pos, val_neg, test, id2word,word2id, vocab_to_consider= pickle.load( open("SobKisu.pkl", "rb"))
import winsound
# Load the important stuff
import numpy as np
import pickle
import pandas as pd
import time
import math

#Read Training, validation and test sets
Xtrain, ytrain, Xval, yval, Xtest, ytest = pickle.load(open("Important.pk", "rb"))
#pickle.dump((df,y_train, df_val, y_val,df_test, y_test), open("Important.pk", "wb"))
#Xtrain, ytrain, Xval, yval, Xtest, ytest = pickle.load(open("IMDBProcessed.pkl", "br"))

#Reshape all the training sets as matrix or vectors
Xtrain = Xtrain.values
Xval = Xval.values
Xtest = Xtest.values
ytrain = np.asarray(ytrain)
ytrain = np.reshape(ytrain, (20000, 1))

yval = np.asarray(yval)
yval = np.reshape(yval, (5000, 1))

ytest = np.asarray(ytest)
ytest = np.reshape(ytest, (25000, 1))

#initialize variables
theta = np.random.uniform(-0.5, 0.5, 2000)  # vector of dimention d, i.e. 2000
theta = np.reshape(theta, (2000, 1))  # 2000X1
theta0 = np.random.uniform(-0.5, 0.5, 1)  # scalar
theta0 = np.reshape(theta0, (1, 1))
alpha = 0.1

best_accuracy = 0
best_theta0 = theta0
best_theta = theta


names=["epoch","Validation Accuracy","Training Accuracy"]

df=pd.DataFrame(columns=names)

#sigmoid function
def sigmoid(X, theta, theta0):
    z = np.array(np.dot(X, theta), dtype=np.float32) + theta0

    return 1.0 / (1.0 + np.exp(-z))


for epoch in range(0, 300):  # count=0
    print(epoch)
    for j in range(0,20000,20):
        #Prepare minibatch
        Xbatch = Xtrain[j:j + 20]  # 20  X 2000
        pred = sigmoid(Xbatch, theta, theta0)
        # calculate error
        error = pred - ytrain[j:j + 20]
        # calculate gradient
        grad = Xbatch.T.dot(error)
        # update theta values
        theta0 = theta0 - .05*alpha * error.sum()
        theta = theta - .05*alpha * grad
    yhat = np.where(sigmoid(Xval, theta, theta0) >= 0.5, 1, 0)
    vaccuracy = np.sum(np.where(yhat == yval, 1, 0)) / float(len(yval)) * 100
    yhat = np.where(sigmoid(Xtrain, theta, theta0) >= 0.5, 1, 0)
    accuracy = np.sum(np.where(yhat == ytrain, 1, 0))/ float(len(ytrain)) * 100
    ajaira=[epoch,vaccuracy,accuracy]

    df=df.append(pd.DataFrame([ajaira], columns=names))


    if (vaccuracy > best_accuracy):
        best_accuracy = vaccuracy
        best_theta = theta
        best_theta0 = theta0
        print('Best Accuracy: ')
        print(vaccuracy)
        print('Best Theta: ')
        print(theta)
        print('Best Theta0: ')
        print(theta0)

print('Best Theta is:')
print(best_theta)
print('Best Theta0 is:')
print (best_theta0)
#Dump model parameters and the best validationa accuracy
pickle.dump((best_theta, best_theta0,best_accuracy), open("results.pk", "wb"))

#get predictions on test set
yhat = np.where(sigmoid(Xtest, best_theta, best_theta0) >= 0.5, 1, 0)

#Calculate test accuracy
test_accuracy = np.sum(np.where(yhat == ytest, 1, 0))/ float(len(ytest)) * 100
print('Test accuracy is:')
print(test_accuracy)
#Dump test accuracy, learning curve as a csv

pickle.dump((test_accuracy), open("TestAccuracy.pk", "wb"))
pickle.dump((df), open("LearningCurve.pk", "wb"))
export_csv = df.to_csv ('LearningCurve.csv', index = None, header=True)
