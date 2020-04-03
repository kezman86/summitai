import csv

import json
import multiprocessing.pool as Pool
import os

import requests
from cryptography.fernet import Fernet
from jira import JIRA
from pandas import DataFrame
from simple_salesforce import Salesforce

from pymongo import MongoClient
import logging

import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


from datetime import datetime
#

import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

#



def get_next_line ( filename ) :
    """ function to return the nex line from file"""
    with open ( filename , encoding = "utf8" ) as csv_file :
        csv_reader = csv.reader ( csv_file , delimiter = ',' )
        for line in csv_reader :
            yield line


def get_next_case( caseslist , n):
    for i in range(0, len(caseslist), n):
        yield caseslist[i:i + n]


class SummitAIMail:
    def __init__( self, config ):
       self.smtp = smtplib.SMTP ( config['smtp_server'] )
       self.msg = ""

    def sendmail( self, to, fr, msg ):
        self.smtp.sendmail ( fr, to , msg )



class SummitAIJIRA :
    """ Class for JIRA connection and operations """

    def __init__ ( self , config ) :
        self.options = {'server' : config["server"] , 'verify' : False}
        self.basicauth = (config["username"] , config["password"])

        # print("Options:{}".format(self.options))
        # print ( "basicauth:{}".format ( self.basicauth ) )

        self.jira = JIRA ( options = self.options , basic_auth = self.basicauth )

    def getJIRAComments ( self , jiraID ) :
        issue = self.jira.issue ( jiraID )
        return (issue.raw['fields']['comment']['comments'])

    def getJIRAFixDetails( self, jiraID ):
        issue = self.jira.issue(jiraID)
        resolution = issue.raw['fields']['resolution']['name']
        status = issue.raw['fields']['status']['name']
        fixDetails = issue.raw['fields']['customfield_10120']
        fixCustImpact = issue.raw['fields']['customfield_10203']
        fixRootCause = issue.raw['fields']['customfield_14900']

        return {
            'jiraResolution': resolution,
            'jiraStatus':status,
            'fixDetails':fixDetails,
            'fixImpact': fixCustImpact,
            'fixRootCasue':fixRootCause
        }


class SummitAIDB :
    def __init__ ( self , options, cloud) :
        print ( "Connecting to MongoDB   ..................." )
        if cloud is not None:
            link = "mongodb+srv://{}:{}@{}/test?retryWrites=true&w=majority".format(options['dbuser'] , options['dbpass'] , options['dbhost'])
        else:
            link = 'mongodb://{}:{}@{}/{}'.format ( options['dbuser'] , options['dbpass'] , options['dbhost'] ,
                                                options['dbname'] )
        try:
            self.dbclient = MongoClient ( link )
        except:
            print ("Connection failed")

        ret = self.dbclient.test
        print (ret)

    def createSetup ( self ) :
        """
        Create database and collections
        """
        pass

    def getDBClient ( self ) :
        return self.dbclient

    def testConnection ( self ) :
        print ( self.dbclient.server_info ( ) )


class SummitAISalesforceLogin :
    SALESFORCECASESFILE = 'summitai/data/sdfccases_new.csv'
    SALESFORCOMMENTSFILE = 'summitai/data/sdfccomments.csv'
    SALESFORCOMMENTSFILE2 = 'summitai/data/test2.csv'
    SALESFORCE_PASSWORD = ''
    SALESFORCE_REPORT_ENDPOINT = "https://misys.my.salesforce.com/00O0J000008Ikeg?export=1&enc=UTF-8&xf=csv"
    SALESFORCE_CASE_ENDPOINT = "https://misys.my.salesforce.com/services/data/v47.0/sobjects/Case/"
    SALESFORCE_CASE_COMMENTS_ENDPOINT = "https://misys.my.salesforce.com/services/data/v47.0/sobjects/Case/{}/CaseComments"

    def __init__ ( self , loginoptions, summitaicrypt ) :
        self.username = loginoptions['username']
        self.password = summitaicrypt.decrypt(loginoptions['password'].encode())
        self.password = self.password.decode().strip()

        #self.password = loginoptions['password']
        self.security_token = loginoptions['security_token']
        self.sf = None
        self.headers = None
        self.sid = None

    def connect ( self ) :
        print ( "Connecting to Salesforce ........" )
        self.sf = Salesforce ( self.username ,
                               self.password ,
                               self.security_token
                               )
        self.headers = self.sf.headers
        self.sid = self.sf.session_id
        print ( "Connected  to Salesforce ........ OK " )


    def getCaseComments ( self , caseId ) :
        with requests.session ( ) as s :
            d = s.get ( self.SALESFORCE_CASE_COMMENTS_ENDPOINT.format ( caseId ) ,
                        headers = self.headers ,
                        cookies = {'sid' : self.sid} )
        return d.text

    def getCases( self ):
        casess = self.sf.query("SELECT ID, Subject, Description  from Case where OwnerId = '00G20000001SEBuEAO' AND Status = 'New' ")
        #case = self.sf.Case.get ( caseID )
        cases = casess['records']

        casesList = []
        for c in cases:
            casesList.append({'caseID' : c['Id'] , 'subject' : c['Subject'], 'description' : c['Description']})

        return casesList



class SummitLogger :
    def __init__ ( self , filename ) :
        logging.basicConfig ( filename = filename ,
                              filemode = 'a' ,
                              format = '%(asctime)s,%(msecs)d %(levelname)s %(message)s' ,
                              datefmt = '%H:%M:%S' ,
                              level = logging.INFO )

        self.logger = logging.getLogger ( )

    def getlogger ( self ) :
        return self.logger

    #TO DO

class SummitAIAppCofig :
    def __init__ ( self , configfile ) :
        with open ( configfile ) as fp :
            loaded_json = json.load ( fp )
            appconfig = loaded_json['appconfig']

            self.database = appconfig['database']
            self.salesforce = appconfig['salesforce']
            self.jiras = appconfig['jira']
            self.logger = appconfig['logger']
            self.mail = appconfig['mail']


class SummitAISalesforceCase :
    """"
        Class for Salesforce case
    """

    def __init__ ( self , infos ) :
        self.caseID = infos[0]
        self.caseNumber = infos[1]

        self.parentId = infos[3]
        self.parentNum = infos[4]

        self.caseSubject = infos[5]
        self.caseDescription = infos[6]
        self.caseStepsToReproduce = infos[7]
        self.caseOpenDate = infos[8]

        self.caseAge = int ( infos[9] )

        self.caseStatus = self.getCaseStatus ( infos[10] )

        self.caseModule = infos[12]
        self.caseSUMVersion = infos[13]
        self.caseAccountName = infos[14]

        self.caseSeverity = self.getCaseSeverity ( infos[15] )
        self.casePriority = self.getCasePriority ( infos[16] )

        self.caseL2Location = infos[17]
        self.caseJIRAID = ""
        self.caseIsBug = '0'
        self.comments = []

    def getCaseStatus ( self , OpenValue ) :
        if OpenValue == '0' :
            return 'c'
        else :
            return 'o'

    def getCaseSeverity ( self , severity ) :
        if 'High' in severity :
            return 'h'
        elif 'Critical' in severity :
            return 'c'
        elif 'Medium' in severity :
            return 'm'
        else :
            return 'l'

    def getCasePriority ( self , severity ) :
        if 'High' in severity :
            return 'h'
        elif 'Critical' in severity :
            return 'c'
        elif 'Medium' in severity :
            return 'm'
        elif 'Showstopper' in severity :
            return 's'
        elif 'Blocker' in severity :
            return 'b'
        else :
            return 'l'

    def toDict ( self ) :
        return {
            'caseID' : self.caseID ,
            'caseNumber' : self.caseNumber ,
            'parentId' : self.parentId ,
            'parentNum' : self.parentNum ,
            'caseSubject' : self.caseSubject ,
            'caseDescription' : self.caseDescription ,
            'caseStepsToReproduce' : self.caseStepsToReproduce ,
            'caseOpenDate' : self.caseOpenDate ,
            'caseAge' : self.caseAge ,
            'caseStatus' : self.caseStatus ,
            'caseModule' : self.caseModule ,
            'caseSUMVersion' : self.caseSUMVersion ,
            'caseAccountName' : self.caseAccountName ,
            'caseSeverity' : self.caseSeverity ,
            'casePriority' : self.casePriority ,
            'caseL2Location' : self.caseL2Location ,
            'caseJIRAID' : self.caseJIRAID ,
            'caseIsBug' : self.caseIsBug ,
            'comments' : self.comments
        }


class SalesforceComment :
    def __init__ ( self , infos ) :
        self.caseID = infos[0]
        self.commentBody = infos[2]
        self.commentCreatedAt = infos[3]

    def toDict ( self ) :
        return {
            'caseID' : self.caseID ,
            'Body' : self.commentBody ,
            'createdAt' : self.commentCreatedAt
        }


class SummitAICrypt:
    SUMMIT_CRYPT_PUB_KEY_PATH = "key.pub"
    def __init__(self):
        if not os.path.isfile ( self.SUMMIT_CRYPT_PUB_KEY_PATH ):
            print ( "Public key doens't exists, generate and save it")
            self.generate_key()
        print("Create crypt")
        self.key = self.load_key()
        self.crypt = Fernet ( self.key )


    def generate_key (self) :
        """
        Generates a key and save it into a file
        """
        self.key = Fernet.generate_key ( )
        with open ( self.SUMMIT_CRYPT_PUB_KEY_PATH , "wb" ) as key_file :
            key_file.write ( self.key )


    def load_key (self) :
        """
        Loads the key from the current directory named `key.key`
        """
        return open ( self.SUMMIT_CRYPT_PUB_KEY_PATH , "rb" ).read ( )


    def encrypt ( self, word  ) :
        """
        Given a str and key (bytes), it encrypts the file and write it
        """
        return  self.crypt.encrypt ( word )


    def decrypt (self,  word ) :
        """
        Given a str it decrypts it
        """
        return  self.crypt.decrypt ( word )



class SummitAIApp :
    def __init__ ( self , config ) :
        self.appconfig = SummitAIAppCofig ( config )
        self.summitaicrypt = SummitAICrypt ( )

        #self.sf = SummitAISalesforceLogin ( self.appconfig.salesforce, self.summitaicrypt )
        #self.sf.connect()

        self.db = SummitAIDB ( self.appconfig.database , cloud = True)
        #self.db.testConnection()

        #self.jira = SummitAIJIRA ( self.appconfig.jiras , self.summitaicrypt)


        #self.mail = SummitAIMail(self.appconfig.mail)

    def saveCase ( self , obj ) :
        line = obj['line']
        dbcon = obj['dbcon']

        col = dbcon['newai']['salesforcecases']
        newCase = SummitAISalesforceCase ( line )

        if line[2] != '' :
            print ( "Skip parent case" )
            return

        key = {'caseId' : newCase.caseID}
        data = newCase.toDict ( )
        col.update ( key , data , upsert = True )

        # col.insert_one(newCase.toDict())

    def loadCasesFromFile ( self ) :
        f = get_next_line ( "data/cases.csv" )
        t = Pool.ThreadPool ( 24 )
        for i in f :
            obj = {'line' : i , "dbcon" : self.db.getDBClient()}
            t.apply_async ( self.saveCase , (obj ,) )
        t.close ( )
        t.join ( )
        return


    def loadJira ( self, obj ) :
        """ function used to save a case in db """
        line = obj['line']
        dbcon = obj['dbcon']

        caseNumber = line[2]
        jiraID = line[18]
        if jiraID == '':
            return

        col = dbcon['newai']['salesforcecases']

        # if it has Customer Case ID => it is parrent case, so we don't save it to db
        if caseNumber :
            col.update_one ( {'caseNumber' : caseNumber},{"$set": {'caseJIRAID' : jiraID}} )

        return

    def populateJIRA ( self ) :
        f = get_next_line ( "data/cases.csv" )
        t = Pool.ThreadPool ( 24 )
        for i in f :
            obj = {'line' : i , "dbcon" : self.db.getDBClient()}
            t.apply_async ( self.loadJira , (obj ,) )
        t.close ( )
        t.join ( )

    def saveComment (self, obj ):

        line = obj['line']
        dbcon = obj['dbcon']

        col = dbcon['newai']['salesforcecases']

        if col == None :
            print ( "Error db connection!" )

        newCom = SalesforceComment ( line )

        if line[1] != '':
            # parent case / save comment under customer case
            tempCase = dbcon['newai']['salesforcecases'].find_one ( {'caseNumber' : line[1]} )

            if tempCase is not None :
                newCom.caseID = tempCase['caseID']

        # print("set comment on customer case {}".format(newCom.caseID))
       # print("Insert comment to {} - {} ".format(newCom.caseID , newCom.toDict()))
        col.update_one ( {'caseID' : newCom.caseID}, {"$push" : { 'comments': newCom.toDict ( )}} )

    def loadCommentFromFile ( self ) :
        f = get_next_line ( "data/comments.csv" )
        t = Pool.ThreadPool ( 24 )
        for i in f :
            obj = {'line' : i , "dbcon" : self.db.getDBClient()}
            t.apply_async ( self.saveComment , (obj ,) )
        t.close ( )
        t.join ( )

        return

    def addFixDetails( self , obj ):
        case = obj['case']
        dbcon = self.db.getDBClient()

        col = dbcon['newai']['salesforcecases']

        if col == None :
            print ( "Error db connection!" )
            return

        jiraID = case[0]['caseJIRAID']
        _id  = case[0]['_id']

        print ('Jira {}'.format(jiraID))



        fixDetails = self.jira.getJIRAFixDetails(jiraID)

        print("Setting fix details")

        col.update_one ( {'_id' : _id} , {"$set":{'fixdetails':fixDetails}} )




    def updateCasesWithJira( self ):
        dbcon =  self.db.getDBClient ( )
        col = dbcon['newai']['salesforcecases']


        casesWithJira = col.find( {'caseJIRAID' : {'$ne' : ''} }, { '_id': 1, 'caseJIRAID': 1 } )

        listCasesWithJira = list(casesWithJira)

        cases = get_next_case(listCasesWithJira, 1)

        t = Pool.ThreadPool ( 24 )

        for i in cases :
            obj = {'case' : i }
            t.apply_async ( self.addFixDetails , (obj ,) )
        t.close ( )
        t.join ( )
        return

    def checkCases( self ):
        '''
            1. Load cases from table newcases
            2. load new cases from Summit Support queue
            3. merge the list -> insert new ones and send mails with them
        '''

        #1
        dbcl =  self.db.getDBClient()

        dbcases = dbcl['newai']['newcases']

        allNewCases = list ( dbcases.find ( {} , {'caseID' : 1 , 'subject' : 1} ) )

        newCasesToSend = []
        cases = self.sf.getCases()

        #check here if any old cases was removed from the queue
        for case in allNewCases:
           c = {'caseID': case['caseID'] , 'subject' : case['subject']}
           if c not in cases:
               dbcases.delete_one({'caseID' : c['caseID']})

        allNewCases = list ( dbcases.find ( {} , {'caseID' : 1 , 'subject' : 1} ) )

        for case in cases:
            id = case['caseID']
            sub = case['subject']

            tempCase = dbcases.find_one ( {'caseID' : id} )

            if tempCase is not None :
                #already exists -> skip
                pass
            else:
                newCasesToSend.append ( case )
                # insert intodb
                key = {'caseID' : id, 'subject': sub}
                dbcases.insert_one ( key )

        linkCase = 'https://misys.my.salesforce.com/{}'




        html = "<h1>Avem in lista <b>{} </b> cazuri noi ( Status = New )</h1>".format(len(newCasesToSend))

        html += "<p> New Cases Since Last Check</p>"
        html += "<ol>"
        for case in newCasesToSend:
            id = case['caseID']
            sub = case['subject']

            linkCase = 'https://misys.my.salesforce.com/{}'.format(id)
            html += "<li><b>" + sub + "   <a href={}>".format(linkCase) + id + "</a></b></li>"

        html += "</ol>"

        html += "<p> New Cases</p>"
        html += "<ol>"


        for case in allNewCases:
            (id , sub) = (case['caseID'] , case['subject'] )
            linkCase = 'https://misys.my.salesforce.com/{}'.format ( id )
            html += "<li><b>" + sub + "   <a href={}>".format ( linkCase ) + id + "</a></b></li>"

        html += "</ol>"

        recipients = ['gheorghe-adrian.cismaru@finastra.com' ]

        for rec in recipients :
            msg = MIMEMultipart ( 'alternative' )

            msg['Subject'] = "Alert Summit Support Cases"
            msg['To'] = rec
            msg['From'] = 'AISummit@finastra.com'

            part2 = MIMEText ( html , 'html' )
            msg.attach ( part2 )

            self.mail.sendmail(rec, "AISummit@finastra.com", msg.as_string())
            print("Mail sent to {}".format(rec))

    def getAllProducts( self ):
        dbcl =  self.db.getDBClient()

        dbcases = dbcl['newai']['salesforcecases']

        allNewCases = list ( dbcases.find ( {} , {'caseModule' : 1 } ) )
        allNewCasesList = []
        for res in allNewCases:
            allNewCasesList.append(res['caseModule'])

        unique_vals = set(allNewCasesList)

        return list(unique_vals)

    def getDataForAI( self ):
        dbcl = self.db.getDBClient ( )

        dbcases = dbcl['newai']['salesforcecases']

        allNewCases = list( dbcases.find ( {} , {'caseNumber': 1, 'caseSubject' : 1 , 'caseDescription' : 1, 'caseModule' : 1} ))

        allc = []
        for c in allNewCases:
            allc.append([c['caseNumber'], c['caseSubject'] + c['caseDescription'], c['caseModule'] ])

        with open ( 'data.json' , 'w' ) as f :
                json.dump ( allc , f )


    def getMainModule( self , sumModule):
        sumModules = {
            "arch" : ["Summit - System level" , "Summit - DB Architecture/Interfaces" ,
                      "Summit - User Interface / FT Framework" , "Summit - DB Admin / Upgrade" ,
                      "Summit - Extendibility / Meta Data" , "Summit - RT Server Architecture And Admin" ,
                      "Summit - Reporting Framework" , "Summit - Performance" ,
                      "Summit - User Documentation" , "Summit - Utilities/Static Data" ,
                      "Summit - APAC - Documentation" , "ARC_License"] ,
            "fo" : ["Summit - Toolkit/Pricing Models" , "Summit - OTC Derivatives" , "Summit - Equities" ,
                    "Summit - Fixed Income" , "Summit - Bond Issuance" ,
                    "Summit - Cash Management/Settlement Processing" ,
                    "Summit - RTF/MKTDATA" , "Summit - Toolkit/Pricing Models" , "Summit - Treasury" ,
                    "Summit - Fixed Income" , "Summit - Structured Products/MUST" ,
                    "Summit - Equity/Equity Derivatives"] ,
            "risk" : ["Summit - Market Risk Limits" , "Summit - Credit risk" , "Summit - Historical VAR" ,
                      "Summit - CCP" ,
                      "Summit - GAP Analysis / Cash Analysis" , "Summit - APAC - Credit Limit" ,
                      "Summit - Risk Management / RT Risk" , "Summit - Credit Derivatives"] ,
            "bo" : ["Summit - Buyside" , "Summit - GBO" ,
                    "Summit - Operations" , "Summit - Commercial Lending" ,
                    "Summit - Collateral Management" , "Summit - Security Finance" ,
                    "Summit - Post Trade Booking Processing" , "Summit - Back Office Positioning" ,
                    "Summit - Back Office Documentation Engine" ,
                    "Summit - Operations/Documentation" , "Summit - Security / STD"] ,
            "acct" : ["Summit - Accounting" , "Summit - P&L" , "Summit - Hedge Accounting/FAS133"]
        }

        if sumModule in sumModules["arch"]:
            return "arch"
        if sumModule in sumModules["risk"] :
            return "risk"
        if sumModule in sumModules["fo"] :
            return "fo"
        if sumModule in sumModules["bo"] :
            return "bo"
        if sumModule in sumModules["acct"] :
            return "acct"
        return None


    def getDataForAI2( self ):
        dbcl = self.db.getDBClient ( )

        dbcases = dbcl['newai']['salesforcecases']

        allNewCases = list( dbcases.find ( {} , {'caseNumber': 1, 'caseSubject' : 1 , 'caseDescription' : 1, 'caseModule' : 1} ))

        allc = []
        for c in allNewCases:
            allc.append([c['caseNumber'], c['caseSubject'] + c['caseDescription'], self.getMainModule(c['caseModule']) ])

        with open ( 'data3.json' , 'w' ) as f :
                json.dump ( allc , f )


    def updateDateTime( self ):
        dbcl = self.db.getDBClient ( )
        dbcases = dbcl['newai']['salesforcecases']

        allNewCases = list (dbcases.find ( {} , {'caseOpenDate' : 1 } ) )

        for c in allNewCases:
            dbcases.update_one ( {"_id" : c['_id']} , {"$set" : {'caseOpenDate' : datetime.strptime(c['caseOpenDate'], "%d/%m/%Y %H:%M")}} )



    def getDataForAI3( self, outputfile, datefrom, dateto ):
        dbcl = self.db.getDBClient ( )
        dbcases = dbcl['newai']['salesforcecases']

        startdate = datetime.strptime(datefrom, '%Y-%m-%d')
        enddate = datetime.strptime(dateto, '%Y-%m-%d')

        allNewCases = list( dbcases.find ( {'caseOpenDate' : {'$gte' : startdate, '$lte' : enddate}} , {'caseNumber': 1, 'caseSubject' : 1 , 'caseDescription' : 1, 'caseModule' : 1} ))

        allc = []
        for c in allNewCases:
            allc.append([c['caseNumber'], c['caseSubject'] + ' ' + c['caseDescription'], self.getMainModule(c['caseModule']) ])

        with open (outputfile , 'w' ) as f :
                json.dump ( allc , f )



    def printNewCases( self ):
        cases = self.sf.getCases ( )
        newCases = []
        for case in cases:
            id = case['caseID']
            sub = case['subject']
            desc = case['description']
            newCases.append([id, sub, sub + " " + desc] )
        return DataFrame(newCases)


    def loadCasesFromCSV( self, filename ):
        df = pd.read_csv (  filename )
        df = df[pd.notnull ( df['ProductModule'] )]
        df = df[pd.isnull(df['CustomerCaseNumber'])]

        dbcl = self.db.dbclient

        dbcases = dbcl['summitai']['cases']


        for index, row in df.iterrows():
            openedDate = datetime.strptime ( row['OpenedDate'] , '%d/%m/%Y' )
            key = {'caseId' : row['CaseID']}

            data = {'caseId' : row['CaseID'], 'parentCaseID' : row['ParentCaseID'], 'customerCaseNumber' : row['CustomerCaseNumber'],
                    'caseNumber': row['CaseNumber'], 'subject' : row['Subject'], 'description' :row['Description'] ,
                    'openedDate' : openedDate, 'productModule' : row['ProductModule']
                    }

            dbcases.update ( key , data , upsert = True )




    def modifyAcct( self, model ):
        dbcases = self.db.dbclient['summitai']['cases']
        allNewCases = list ( dbcases.find (
            {'openedDate' : {'$gte' : model.dateFrom , '$lte' : model.dateTo} ,
             'productModule' : "Summit - Accounting, P&L and FAS"} ,
            {'caseNumber' : 1 , 'subject' : 1 , 'description' : 1 , 'productModule' : 1} ) )

        allc = []
        toPreditcX = []
        for c in allNewCases :
            subPlusDesc = c['subject'] + ' ' + c['description']
            if c['productModule'] is not None and subPlusDesc is not None :
                toPreditcX.append ( subPlusDesc )

        toPreditcX = pd.Series ( toPreditcX )
        pred = model.predict(toPreditcX)

        for i in range(toPreditcX.size ):
            dbcases.update_one ( {'caseNumber' : allNewCases[i]['caseNumber']} , {"$set" : {'productModule' : pred[i]}} )

















