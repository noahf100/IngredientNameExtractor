import tensorflow as tf
import numpy as np
import sys
import pickle
import os
import gc
from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = ''

args = sys.argv
if len(args) != 3:
    print(len(args))
    print("Arguement mismatch. Proper input = training-doc-path testing-doc-pat\
h")
    sys.exit(0)

TRAINING_DOC = args[1]
TESTING_DOC = args[2]

#data process
def process(path, wordDict, resDict, charDict):
    contents = ""
    with open(path, 'r') as f:
        contents = f.read()

    lines = contents.split('\n')[1:]
    words = [l.split()[0].lower() for l in lines if len(l) > 0]
    labels = [l.split()[-1] for l in lines if len(l) > 0]
    noSpaces = "".join(contents.split())

    unique = []
    labs = []

    unique = list(set(words))
    unique.append("STOP")
    unique.append("UNKNOWN")
    labs = list(set(labels))
    labs.append("STOP")

    vocab_size = len(unique)
    label_size = len(labs)

    dict = {}
    resDic = {}
    charDic = {}
    
    if wordDict is not None and resDict is not None and charDict is None:
        dict = wordDict
        resDic = resDict
        charDic = charDict
    else:
        for i in range(vocab_size):
            dict[unique[i]] = i

        for i in range(label_size):
            resDic[labs[i]] = i

        counter = 0
        for i in range(len(noSpaces)):
            c = noSpaces[i:i + 1]
            if c not in charDic:
                charDic[c] = counter
                counter += 1
        

    sentence = []
    sentences = []
    maxV = 0

    maxC = 0
    for w in words:
        maxC = max(maxC, len(w))

    for l in lines:
        if len(l) < 1:
            sentences.append(sentence)
            maxV = max(maxV, len(sentence))
            sentence = []
            continue
        sentence.append(l)
        
    sentences = sentences[1:-1]

    data = np.zeros([len(sentences), maxV], dtype=np.int32)
    dataLabels = np.zeros([len(sentences), maxV], dtype=np.int32)

    chars = np.zeros([len(sentences), maxV, maxC], dtype=np.int32)
    wordLengths = np.zeros([len(sentences), maxV], dtype=np.int32)

    print(maxV)

    for i in range(len(sentences)):
        for j in range(maxV):
            if j < len(sentences[i]):
                word = sentences[i][j].split()[0]
                try:
                    data[i][j] = dict[word.lower()]
                    dataLabels[i][j] = resDic[sentences[i][j].split()[-1]]
                except KeyError:
                    data[i][j] = dict["UNKNOWN"]
                    dataLabels[i][j] = resDic["O"]

                try:
                    wordLengths[i][j] = len(word)
                    for k in range(len(word)):
                        chars[i][j][k] = charDic[word[k:k+1]]
                except KeyError: 
                    wordLengths[i][j] = -1
                    #for k in range(maxC):
                    #    chars[i][j][k] = -1
            else:
                data[i][j] = dict["STOP"]
                dataLabels[i][j] = resDic["STOP"]
                wordLengths[i][j] = -1
                #for k in range(maxC):
                #    chars[i][j][k] = -1
                
    print(sentences[0])
    print(wordLengths[0])
    return dict, resDic, charDic, vocab_size, label_size, data, dataLabels, wordLengths, chars

dict, resDict, charDict, vocab_size, label_size, data, dataLabels, wordLengths, wordLabels = process(TRAINING_DOC, None, None, None)
_, _, _, _, _, data_test, dataLabels_test, wordLengths_test, wordLabels_test = process(TESTING_DOC, dict, resDict, charDict)

gc.collect()

pickle.dump(dict, open("models/dict.pkl", "wb" ))
pickle.dump(resDict, open("models/resDict.pkl", "wb" ))
pickle.dump(charDict, open("models/charDict.pkl", "wb" ))

model = Model(vocab_size, len(charDict.keys()))
model.build()
model.train(data, dict, dataLabels, data_test, resDict, dataLabels_test, charDict, wordLengths, wordLabels, wordLengths_test, wordLabels_test)
pickle.dump(dict, open("models/dict.pkl", "wb" ))
pickle.dump(resDict, open("models/resDict.pkl", "wb" ))

