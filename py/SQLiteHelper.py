import sqlite3 as sqlite
import sys,os,re
from os import listdir
from os.path import isfile, join
from csv import reader


def escapeSpecialCharacters ( text, characters ):
    for character in characters:
        text = text.replace( character, "'" + character )
    return text

class SQLiteHandler:		
	def __init__(self, dbfile):
		self.conn=sqlite.connect(dbfile)

	def __del__(self):
		self.conn.close()
		#print "### Closed db connection"

	def dbcommit(self):
		self.conn.commit()	
	
 
	def run_sql(self,sql):
		res=self.conn.execute(sql)
		return res 		