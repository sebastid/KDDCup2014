from __future__ import division
import datetime
import sys,io
import pprint,pickle
import itertools
import numpy as np
import pandas as pd
from sklearn import (metrics, cross_validation, linear_model, naive_bayes, preprocessing)
from collections import defaultdict

sys.path.append('/Users/sebastiendurand/Documents/Kaggle/KDDCup2014/py')


SEED = 42  # always use a seed for randomized procedures
CSV_DIR = "/Users/sebastiendurand/Documents/Kaggle/KDDCup2014/csv/"


def SaveResults(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def SaveResults2(df,VERSION):
    filename = '../csv/submission_%s_%s.csv' % (VERSION,datetime.date.today())
    df.to_csv(filename,sep=',',index=False)
    print "\nSaved %d rows file: %s" % (len(df),filename)
    
from SQLiteHelper import SQLiteHandler
import pandas.io.sql as psql
def LoadSqlData(sql):
	h=SQLiteHandler("kaggle.db")
	df = psql.frame_query(sql, h.conn)
	return df

#------------ Naive Bayes Training
import re
def ExtractVocabulary(D):
    V = {}
    for doc in D["short_description"]:
        tokens = re.split("\W",doc)
        tokens = filter(lambda x: len(x)>4,tokens)
        for t in tokens:
            t = t.lower()
            if V.has_key(t):
                V[t]+=1
            else:
                V[t]=1
    return V

def CountDocsInClass(D,c):
    return D[D["is_exciting"]==c].count()[1]

def ConcatenateTextOfAllDocsInClass(D,c):
    index = {}
    class_docs = D[D["is_exciting"]==c]
    V = ExtractVocabulary(class_docs)
    for k,v in V.items():
        word = k.lower()
        if not index.has_key(word):
            index[word] = v
        else:
            index[word] += v
    return index
    #return ''.join(D[D["is_exciting"]==c]["short_description"].values).lower()

def CountTokensOfTerm(textc, t):
    if not textc.has_key(t):
        return 0
    return textc[t]
    
def TrainMultinomial(C,D):
    V = ExtractVocabulary(D)
    N = len(D) * 1.0
    prior = {}
    condprob = {}
    print " Extracted Vocabulary Size: ",len(V)
    for c in C:
        NC = CountDocsInClass(D,c)
        prior[c] = NC / N
        textc = ConcatenateTextOfAllDocsInClass(D, c)
        
        print " Class",c,"text concatenated",len(textc)
        Tc = {}
        Tcsum = 0.0
        for t in V.keys():
            Tc[t] = CountTokensOfTerm(textc, t)
            Tcsum += Tc[t]+1
            
        condprob[c] = {}
        for t in V.keys():
            condprob[c][t] = (Tc[t]+1)/Tcsum
            
    return V,prior,condprob

#------------ Naive Bayes Application    
def ExtractTokensFromDoc(V, d):
    tokens = re.split("\W",d)
    tokens = filter(lambda x: len(x)>4,tokens)
    tokens = map(lambda x: x.lower(),tokens)
    res = [w for w in tokens if V.has_key(w)]
    return res

def ApplyMultinomialNB(C,V,condprob,prior,d):
    W = ExtractTokensFromDoc(V, d)
    score = {}
    for c in C:
        score[c] = np.log(prior[c])
        for t in W:
           score[c] += np.log(condprob[c][t])
    return score

def argmax(score):
    return max(score, key=lambda k: score[k])

class MultinomialNB:
    def fit(self,X):
        self.C = np.unique(X.is_exciting).values
        V,prior,condprob = TrainMultinomial(self.C,X)
        self.V = V
        self.prior = prior
        self.condprob = condprob

    def predict(self,X):
        scores = []
        for ix,row in X.iterrows():
            score = ApplyMultinomialNB(self.C,self.V,self.condprob,self.prior,row["short_description"])
            scores.append( argmax(score) )
        return scores

    def predict_proba(self,X):
        scores = []
        i=0
        for ix,row in X.iterrows():
            i+=1
            if i%20000==0: print " predicted %d / %d  docs" % (i,X.count()[0])
            logprobs = ApplyMultinomialNB(self.C,self.V,self.condprob,self.prior,row["short_description"])
            probs = [logprobs['f'], logprobs['t']]
            probs = np.exp(probs)
            probs = probs / np.sum(probs)
            if np.isnan(probs).any():
                print "!!!! probs is nan",probs,i,ix,row
                raise "Exception with nan"
            scores.append( probs[1] )
        return scores
#-----------------

def train(x):   
    model = MultinomialNB()
    model.fit(x)            
    return model

def evaluate(x, models):
    preds_m = []
    for i in range(len(models)):
        print "Evaluation of model %d...." % i
    	preds_m.append(models[i].predict_proba(x))   
    preds = np.mean(preds_m,axis=0)

    if np.isnan(preds).any():
        print "Found some null values in preds",preds
        return preds
    
    y = x['is_exciting']
    if len(np.unique(y=='t'))==2:
        auc = metrics.roc_auc_score(y=='t', preds)
        print "AUC: %f" % (auc)
    return preds


def predict(x,models):
    preds_m = []
    for i in range(len(models)):
        print "Prediction of model %d...." % i
    	preds_m.append(models[i].predict_proba(x))
    
	preds = np.mean(preds_m,axis=0)
    if np.isnan(preds).any():
        print "Found some null values in preds",preds
        return preds
    
    res = {"projectid": [], "is_exciting": []}
    for ix,row in x.iterrows():
        res["projectid"].append(row['projectid'])
        pred = preds[ix]
        res["is_exciting"].append(pred)
    
    return pd.DataFrame(res,columns=['projectid','is_exciting'])

#-----------------

	
def main():
	print "Loading data"
	train_df = LoadSqlData(" \
	select t1.projectid, t2.short_description, t2.essay, t3.is_exciting \
	from projects as t1 inner join \
	essays as t2 on t2.projectid=t1.projectid inner join \
	outcomes as t3 on t3.projectid=t1.projectid \
	where t1.date_posted < '2014-01-01'")
	
	test_df = LoadSqlData(" \
	select t1.projectid, t2.short_description, t2.essay \
	from projects as t1 inner join \
	essays as t2 on t2.projectid=t1.projectid \
	where t1.date_posted >= '2014-01-01'")
	
	print "Training...."
	tot = train_df.count()[0]
	nb = int(tot * 0.025)
	print tot,nb
	indices = {}
	subspace_train_df = {}
	models = {}
	MODEL_NB = 100
	for i in range(MODEL_NB):
		print "Training model %d..." % i		
		indices[i] = np.random.choice(train_df.index,nb,replace=False)
		subspace_train_df[i] = train_df.iloc[indices[i]]
		models[i] = train(subspace_train_df[i])
	
	totind = []
	for li in indices.values():
		totind = np.append(totind, li)
	unique_indices = np.unique(totind)
	print "Used %d out of %d training rows" % (len(unique_indices), train_df.count()[0])
	eval_indices = np.setdiff1d(np.array(train_df.index),unique_indices)
	print "Using %d rows for the ROC evaluation" % len(eval_indices)

	eval_res = evaluate(train_df.iloc[eval_indices],models)

	preds = predict(test_df, models)
	VERSION = '3'
	SaveResults2(preds,VERSION) 
	
if __name__=="__main__":
	main()
	
