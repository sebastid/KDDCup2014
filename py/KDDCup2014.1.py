from __future__ import division
import sys,io
import pprint,pickle
import itertools
import numpy as np
import pandas as pd
from sklearn import (metrics, cross_validation, linear_model, naive_bayes, preprocessing)
from collections import defaultdict


SEED = 42  # always use a seed for randomized procedures
CSV_DIR = "/Users/sebastiendurand/Documents/Kaggle/KDDCup2014/csv/"
DATE = '2014-04-29'
VERSION = '1'

def SaveResults(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

from SQLiteHelper import SQLiteHandler
import pandas.io.sql as psql
def LoadSqlData(sql):
	h=SQLiteHandler("kaggle.db")
	df = psql.frame_query(sql, h.conn)
	return df
	
def main():
	print "Loading data"
	train_df = LoadSqlData('select t1.projectid,t1.short_description, t1.essay,t2.is_exciting from essays as t1 inner join  outcomes as t2 on t2.projectid=t1.projectid')

if __name__=="__main__":
	main()
	
