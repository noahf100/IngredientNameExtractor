import tensorflow as tf
import numpy as np
import sys
import pickle
from model import Model

dict = pickle.load( open( "models/dict.pkl", "rb" ) )
resDict = pickle.load( open( "models/resDict.pkl", "rb" ) )
charDict = pickle.load( open( "models/charDict.pkl", "rb" ) )

revResDict = {}
for i in resDict.keys():
    revResDict[resDict[i]] = i

revDict = {}
for i in dict.keys():
    revDict[dict[i]] = i

model = Model(len(dict.keys()), len(charDict.keys()), len(resDict.keys()))
model.build()
model.restore("./models/model")

while True:
    try:
        # for python 2                                                      
        sentence = raw_input("> ")
    except NameError:
        # for python 3                                                      
        sentence = input("> ")

    # create dataset
    words = sentence.split()
    labs = [0 for i in range(len(words))]
    w = [0 for i in range(len(words))]
    lens = [0 for i in range(len(words))]
    
    maxC = 0

    for i in range(len(words)):
        maxC = max(maxC, len(words[i]))
        lens[i] = len(words[i])
        if words[i].lower() in dict:
            w[i] = dict[words[i].lower()]
        else:
            w[i] = dict["UNKNOWN"]

    chars = np.zeros([1, len(words), maxC])
    for i in range(len(words)):
        for j in range(len(words[i])):
            chars[0][i][j] = charDict[words[i][j]]

    #Run model
    w = np.array(w).reshape(1, -1)
    lens = np.array(lens).reshape(1, -1)
    
    res = model.predict(revResDict, w, lens, chars, [len(words)])

    #Get results
    str1 = ""
    str2 = ""
    for i in range(len(words)):
        str1 += str(words[i]) + "\t"
        str2 += str(res[i]) + "\t"

    print(str1)
    print(str2)
