import sqlite3 as sqlite
import sys,os,re
from os import listdir
from os.path import isfile, join
from csv import reader

#
#- 
#
class CVSFileHandler:
	def __init__(self, cvsfile):
		m = re.match("^.+/(\w+).csv$",cvsfile)
		if m:
			tablename = m.group(1)
			headers = None
			db = None

			i = 0
			print "--------------------"
			print "Reading file %s" % csvfile
			f = open(csvfile,"r")
			for cvals in reader(f):
				if i==0:
					headers=','.join(cvals)
					db = SQLiteHandler("kaggle.db",tablename,cvals)			
				else:					
					db.insert(tablename,headers,cvals)
					value = None		
				i += 1
				if i%10000==0: print "Processed %d rows" % i
			db.dbcommit()
			print "Inserted %d rows" % (i+1)
			print ""
			f.close()

def escapeSpecialCharacters ( text, characters ):
    for character in characters:
        text = text.replace( character, "'" + character )
    return text

class SQLiteHandler:		
	def __init__(self, dbfile, tablename, columnsarr):
		self.conn=sqlite.connect(dbfile)
		
		#if columnsarr[0]!="id": 
		#	columnsarr.insert(0,'id')
			
		print 'Header: '+','.join(columnsarr)
		keystr = columnsarr[0]
		columns = ','.join(columnsarr[1:])
		cmd = 'create table if not exists %s(%s primary key, %s, unique(%s))' % (tablename,keystr,columns,keystr)
		print cmd
		self.conn.execute(cmd)
		#print "### Initialized db %s" % dbfile
		self.conn.commit()
		
	def __del__(self):
		self.conn.close()
		#print "### Closed db connection"

		
	def dbcommit(self):
		self.conn.commit()

	def insert(self,tablename,headers,values):
		cmd1 = ""
		try:				
			values_str = ''
			for v in values:
				values_str += "'%s'," % escapeSpecialCharacters(v,r"'")
			
			#cmd1 = 'insert or ignore into %s(%s)  values(%s)' % (tablename,headers,values)
			cmd2 = 'insert into %s(%s)  values(%s)' % (tablename,headers,values_str[:-1])
			cursor = self.conn.cursor()
			cursor.execute(cmd2)
			
		except Exception, ex:
			print cmd1
			print ex
	
	
	def count(self, date):
		res=self.conn.execute('select count(*) from Racecourses where date="%s"' %(date)).fetchone()
		return res 		


if __name__=="__main__":
	print "Starting..."
	if len(sys.argv)>1:
		inst_dir = r"/Users/sebastiendurand/Documents/Kaggle/KDDCup2014/"
		dirs = inst_dir + sys.argv[1]
		onlyfiles = [ f for f in listdir(dirs) if isfile(join(dirs,f)) ]
		for f in onlyfiles:
		    if f.endswith('.csv'):
				csvfile = join(dirs,f)
				print csvfile
				cvshandler = CVSFileHandler(csvfile)
	else:
		print "Usage: csv2sql.py <cvsdir>"