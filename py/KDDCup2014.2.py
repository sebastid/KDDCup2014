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


SEED = 1001
def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=.20, random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.roc_auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N

def train(x):
    X_train = np.matrix(x['essay'].apply(lambda e: len(e))).T
    y = x['is_exciting']
    model = linear_model.LogisticRegression()
    #print cv_loop(X_train,y=='t',model,5)
                  
    model.fit(X_train, y=='t')
    return X_train,y,model

def predict(x,models):
    X_test = np.matrix(x['essay'].apply(lambda e: len(e))).T
    
    preds_m = []
    for i in range(len(models)):
    	preds_m.append(models[i].predict_proba(X_test)[:,1])
    
	preds = np.mean(preds_m,axis=0)
	print preds
    res = {"projectid": [], "is_exciting": []}
    for ix,row in x.iterrows():
        res["projectid"].append(row['projectid'])
        pred = preds[ix]
        res["is_exciting"].append(pred)
    
    return pd.DataFrame(res,columns=['projectid','is_exciting'])
	
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
	
	tot = train_df.count()[0]
	nb = int(tot * 0.1)
	print tot,nb
	indices = {}
	subspace_train_df = {}
	models = {}
	for i in range(1000):
		print "training model",i		
		indices[i] = np.random.choice(np.arange(0,tot),nb,replace=False)
		subspace_train_df[i] = train_df.iloc[indices[i]]
		X,y,models[i] = train(subspace_train_df[i])
	
	
	preds = predict(test_df, models)
	VERSION = '2'
	SaveResults2(preds,VERSION) 
	
if __name__=="__main__":
	main()
	
