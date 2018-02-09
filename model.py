import tensorflow as tf
import numpy as np
import sys
import pickle
import os

class Model:
    def __init__(self, vocab_size, charSize, resSize):
        #vars
        self.lrate = 0.001
        self.epochs = 1
        self.bSize = 1
        self.wordEmbedSize = 128
        self.charEmbedSize = 128
        self.charHiddenSize = 128
        self.lstmSize = 128
        self.dropoutProb = 0.33
        self.vocab_size = vocab_size
        self.char_size = charSize
        self.resSize = resSize

    def addPlaceholders(self):
        self.sequenceLengths = tf.placeholder(tf.int32, shape=[None], name="sequenceLengths")
        self.wordLengths = tf.placeholder(tf.int32, shape=[None, None], name="wordLengths")
        self.input = tf.placeholder(tf.int32, shape=[None, None], name="input")
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None],name="chars")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

    def addWordEmbeddings(self):
        with tf.variable_scope("words"):
            E = tf.Variable(initial_value=tf.random_normal(shape=[self.vocab_size, self.wordEmbedSize], stddev = 0.1), name="E")
            lookup = tf.nn.embedding_lookup(E, self.input, name="lookup")

        with tf.variable_scope("chars"):
            E2 = tf.Variable(initial_value=tf.random_normal(shape=[self.char_size, self.charEmbedSize], stddev = 0.1), name="E2")
            lookup2 = tf.nn.embedding_lookup(E2, self.chars, name="char_lookup")

            s = tf.shape(lookup2)
            char_embeddings = tf.reshape(lookup2,shape=[s[0]*s[1], s[-2], self.charEmbedSize])
            wl = tf.reshape(self.wordLengths, shape=[s[0]*s[1]])

            cell_fw = tf.contrib.rnn.LSTMCell(self.charHiddenSize, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.charHiddenSize, state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings, sequence_length=wl, dtype=tf.float32)

            _, ((_, output_fw), (_, output_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.reshape(output, shape=[s[0], s[1], 2*self.charHiddenSize])

            word_embeddings = tf.concat([lookup, output], axis=-1)

        self.embed = tf.nn.dropout(word_embeddings, self.dropoutProb, name="e_dropout")

    def addLogits(self):
        with tf.variable_scope("bi-lstm"):
            # Forward direction cell
            fw_cell = tf.contrib.rnn.LSTMCell(self.lstmSize)

            # Backward direction cell
            bw_cell = tf.contrib.rnn.LSTMCell(self.lstmSize)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, self.embed, 
            sequence_length=self.sequenceLengths, dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropoutProb)
            

        with tf.variable_scope("proj"):
            W = tf.Variable(initial_value=tf.random_normal(shape=[2*self.lstmSize, self.resSize], stddev=0.1), name="weights")
            b = tf.Variable(initial_value=tf.random_normal(shape=[self.resSize], stddev=0.1), name="bias")

            o1 = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.lstmSize], name="output")
            logits = tf.matmul(output, W) + b
            self.logits = tf.reshape(logits, [-1, o1, self.resSize], name="logits")

    def addLoss(self):
        #Seq2seq
        '''
        softmax = tf.nn.softmax(self.logits)
        self.softmax = tf.cast(tf.argmax(self.logits, 1), tf.int32, name="softmax")

        xEnt = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.sequence_lengths)
        xEnt = tf.boolean_mask(xEnt, mask)

        self.loss = tf.reduce_mean(xEnt, name="loss")
        '''

        #crf
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.sequenceLengths)
        self.trans_params = tf.identity(trans_params, name="trans_params")
        self.loss = tf.reduce_mean(-log_likelihood, name="loss")
        

    def addTraining(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate)
        self.train_op = optimizer.minimize(self.loss, name="train_op")

    def initializeSession(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build(self):
        self.addPlaceholders()
        self.addWordEmbeddings()
        self.addLogits()
        self.addLoss()
        self.addTraining()
        self.initializeSession()


    def train(self, data, dict, dataLabels, data_test, resDict, dataLabels_test, charDict, word_lengths, chars, word_lengths_test, chars_test):
        total_loss = 0.0
        for j in range(self.epochs):
            for i in range(len(data)):
                
                if dict["STOP"] in data[i]:
                    stopIndex = data[i].tolist().index(dict["STOP"])
                    sentence = data[i][:stopIndex]
                    labs = dataLabels[i][:stopIndex]
                    characters = chars[i][:stopIndex]#[:]
                    wordLengths = word_lengths[i][:stopIndex]
                else:
                    sentence = data[i]
                    labs = dataLabels[i]
                    characters = chars[i]
                    wordLengths = word_lengths[i]
                
                #Empty lines
                if stopIndex <= 1:
                    continue

                labs = labs.reshape(1, -1)
                sentence = sentence.reshape(1, -1)
                characters = characters.reshape(1, characters.shape[0], -1)
                wordLengths = wordLengths.reshape(1, -1)
                
                l, _ = self.session.run([self.loss, self.train_op], 
                    feed_dict={
                        self.input: sentence,
                        self.labels: labs,
                        self.sequenceLengths: [stopIndex],
                        self.wordLengths: wordLengths,
                        self.chars: characters
                      })
                total_loss += l
            
                # Print Loss every so often
                if i % 1000 == 0:
                    print 'Iteration %d\tLoss Value: %.3f' % (i, l)

            self.save()
            self.test(data_test, dict, resDict, charDict, dataLabels_test, word_lengths_test, chars_test)


    def test(self, data_test, dict, resDict, charDict, dataLabels_test, word_lengths_test, chars_test):
        total_correct = 0.0
        total_count = 0.0

        revResDict = {}
        for i in resDict.keys():
            revResDict[resDict[i]] = i

        for i in range(len(data_test)):
            sentence = []
            labs = []
            if dict["STOP"] in data_test[i]:
                stopIndex = data_test[i].tolist().index(dict["STOP"])
                sentence = data_test[i][:stopIndex]
                labs = dataLabels_test[i][:stopIndex]
                characters = chars_test[i][:stopIndex]
                wordLengths = word_lengths_test[i][:stopIndex]
            else:
                sentence = data_test[i]
                labs = dataLabels_test[i]
                characters = chars_test[i]
                wordLengths = word_lengths_test[i]
                
            if len(characters) < 1:
                continue

            labs = labs.reshape(1, -1)
            sentence = sentence.reshape(1, -1)
            characters = characters.reshape(1, characters.shape[0], -1)
            wordLengths = wordLengths.reshape(1, -1)

            res = self.predict(revResDict, sentence, wordLengths, characters, [stopIndex])
            
            for l in range(len(res)):
                if res[l] == revResDict[labs[0][l]]:
                    total_correct += 1
                total_count += 1

            if i % 1000 == 0:
                p = total_correct/float(total_count) if total_count > 0 else 0.0
                print 'Iteration %d\tAccuracy: %.3f' % (i, p)

        p = total_correct/float(total_count) if total_count > 0 else 0

        print("Accuracy: " + str(p))

    def restore(self, dir_model):
        print("Restoring Model")
        self.saver.restore(self.session, dir_model)

    def save(self):
        self.saver.save(self.session, 'models/model')

    def createFeedDict(self, w, lens, chars, sequence_lens):
        fd = {
            self.input: w, 
            self.wordLengths: lens, 
            self.chars: chars, 
            self.sequenceLengths: sequence_lens
        }
        return fd

    def predict(self, revResDict, w, lens, chars, seq_l):
        fd = self.createFeedDict(w, lens, chars, seq_l)
        logs, trans_params = self.session.run([self.logits, self.trans_params], feed_dict=fd)

        viterbi_sequences = []

        for logit, sl in zip(logs, seq_l):
            logit = logit[:sl]
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        preds = [revResDict[idx] for idx in list(viterbi_sequences[0])]

        return preds

