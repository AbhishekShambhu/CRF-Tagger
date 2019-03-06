# Conditional Random Field (CRF) Implementation
import nltk
#Using Treebank Corpus
sent_tag = nltk.corpus.treebank.tagged_sents()

#Using basic feature templates 
def feature(sent_word, i):
    return {
        'word': sent_word[i],
        'firstword': i == 0,
        'lastword': i == len(sent_word) - 1,
        'capitalword': sent_word[i][0].upper() == sent_word[i][0],
        'uppercase': sent_word[i].upper() == sent_word[i],
        'lowercase': sent_word[i].lower() == sent_word[i],
        'prev_word': '' if i == 0 else sent_word[i - 1],
        'next_word': '' if i == len(sent_word) - 1 else sent_word[i + 1],
        'prefix1': sent_word[i][0],
        'prefix2': sent_word[i][:2],
        'suffix1': sent_word[i][-1],
        'suffix2': sent_word[i][-2:],        
        'hyphen': '-' in sent_word[i],
        'numeric': sent_word[i].isdigit(),
        'capitals_present': sent_word[i][1:].lower() != sent_word[i][1:]
    }
  
#Building dataset
from nltk.tag.util import untag
 
# Split the dataset for training and testing
cutoff = int(.80 * len(sent_tag))
sent_train = sent_tag[:cutoff]
sent_test = sent_tag[cutoff:]
 
def appenddata(sent_tag):
    X, y = [], [] 
    for tagged in sent_tag:
        X.append([feature(untag(tagged), index) for index in range(len(tagged))])
        y.append([tag for _, tag in tagged]) 
    return X, y
 
X_train, y_train = appenddata(sent_train)
X_test, y_test = appenddata(sent_test)

# importing CRF library
from sklearn_crfsuite import CRF 
crfmodel = CRF()
crfmodel.fit(X_train, y_train)

#making predictions using model
inputsentence = ['We','are','TeamBots','and','this','is','our','simple','implementation']
 
def postagged(inputsentence):
    sentence_features = [feature(inputsentence, index) for index in range(len(inputsentence))]
    return list(zip(inputsentence, crfmodel.predict([sentence_features])[0]))
 
print(postagged(inputsentence))    
# [('We', 'PRP'), ('are', 'VBP'), ('TeamBots', 'NNS'), ('and', 'CC'), ('this', 'DT'), 
# ('is', 'VBZ'), ('our', 'PRP$'), ('simple', 'JJ'), ('implementation', 'NN')]

#computing the performance of the model
from sklearn_crfsuite import metrics
 
y_pred = crfmodel.predict(X_test)
print(metrics.flat_accuracy_score(y_test, y_pred))   # 0.9547382603922352