import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import joblib
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# #############################################################################
# Load some categories from the training set
categories = [
    "acctg",
    "arch",
    "bo",
    "fo",
    "risk"
]
################################################################################

REPLACE_BY_SPACE_RE = re.compile ( '[/(){}\[\]\|@,;]' )
BAD_SYMBOLS_RE = re.compile ( '[^0-9a-z #+_]' )
STOPWORDS = set ( stopwords.words ( 'english' ) )

MODEL_DUMP_EXT = ".pkl"

sumModules = {
    "arch" : ["Summit - System level" , "Summit - DB Architecture/Interfaces" ,
              "Summit - User Interface / FT Framework" , "Summit - DB Admin / Upgrade" ,
              "Summit - Extendibility / Meta Data" , "Summit - RT Server Architecture And Admin" ,
              "Summit - Reporting Framework" , "Summit - Performance" ,
              "Summit - User Documentation" , "Summit - Utilities/Static Data" ] ,
    "fo" : ["Summit - Toolkit/Pricing Models" , "Summit - OTC Derivatives" , "Summit - Equities" ,
            "Summit - Fixed Income" , "Summit - Bond Issuance" , "Summit - Cash Management/Settlement Processing" ,
            "Summit - RTF/MKTDATA" , "Summit - Toolkit/Pricing Models", "Summit - Treasury" , "Summit - Fixed Income","Summit - Structured Products/MUST" , "Summit - Equity/Equity Derivatives"  ] ,
    "risk" : ["Summit - Market Risk Limits" , "Summit - Credit risk" , "Summit - Historical VAR" , "Summit - CCP" ,
              "Summit - GAP Analysis / Cash Analysis" , "Summit - APAC - Credit Limit" ,
              "Summit - Risk Management / RT Risk", "Summit - Credit Derivatives"] ,
    "bo" : ["Summit - Buyside" , "Summit - GBO" ,
            "Summit - Operations" , "Summit - Commercial Lending" ,
            "Summit - Collateral Management" , "Summit - Security Finance" ,
            "Summit - Post Trade Booking Processing" , "Summit - Back Office Positioning" ,
            "Summit - Back Office Documentation Engine" ,
            "Summit - Operations/Documentation" , "Summit - Security / STD"] ,
    "acct" : ["Summit - Accounting", "Summit - P&L", "Summit - Hedge Accounting/FAS133"]
}

def moduleToArea ( sumModule ) :
    if sumModule in sumModules["arch"] :
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


def occurencesCount(dataset):
    dataset = dataset[pd.notnull(dataset['y'])]

    r = dataset['y'].value_counts()

    for idx in r.index:
        if r[idx] < 10:
            dataset = dataset[dataset['y'] != idx]

    return (dataset['X'],dataset['y'])



class SummitAIModel:
    '''
    Interface for AI Models
    '''
    def __init__( self, dbclient, modeldump, dateFrom, dateTo ):
        self.dbclient = dbclient
        self.dateFrom = datetime.strptime ( dateFrom , '%Y-%m-%d' )
        self.dateTo = datetime.strptime ( dateTo , '%Y-%m-%d' )
        self.modeldump = modeldump + MODEL_DUMP_EXT
        self.X = []
        self.y = []
        self.pipe = None

    def clean_text( self, text ):
        ''' function for text cleaning'''
        pass

    def loadData( self ):
        ''' function for loading '''
        pass

    def buildModel( self, transformer, classifier ):
        """
        Train a model with scikit-learn
        :param X: X-train
        :param y: y_train
        :param transformer: transformer class ( TfidVectorizer )
        :param classifier: classifier class , the model
        :param dumpfile: output -> dump model
        """
        summitAIModel = make_pipeline ( transformer , classifier )
        summitAIModel.fit ( self.X , self.y )
        joblib.dump ( summitAIModel , self.modeldump )

    def load_model (self, modeldump):
        pass

    def predict( self, toPredict ):
        pass

    def testModel( self, transformer, classifier  ):
        pass


class SummitAITriageModel(SummitAIModel):
    '''
        Class for AI Model
    '''
    def __init__( self, filename ):
        super(SummitAITriageModel, self).__init__( filename )

    def clean_text ( self, text ) :
        """
        Function to clean the text
        :param text: the text for a case
        :return: the cleaned text
        """
        text = BeautifulSoup ( text , "lxml" ).text  # HTML decoding
        text = text.lower ( )  # lowercase text
        text = REPLACE_BY_SPACE_RE.sub ( ' ' , text )  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub ( '' , text )  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = ' '.join ( word for word in text.split ( ) if word not in STOPWORDS )  # delete stopwors from text
        return text

    def loadData ( self ) :
        """
        Function to get the data for AI model
        :param filename: path to the input file
        :output df[1] - X . df[2] - y
        """
        df = pd.read_json (  self.dataset )
        df = df[pd.notnull ( df[2] )]
        df[1] = df[1].apply ( self.clean_text )

        self.X = df[1]
        self.y = df[2]


    def build_model ( self, transformer, classifier, dumpfile ) :
        """
        Train a model with scikit-learn
        :param X: X-train
        :param y: y_train
        :param transformer: transformer class ( TfidVectorizer )
        :param classifier: classifier class , the model
        :param dumpfile: output -> dump model
        """
        summitAIModel = make_pipeline ( transformer , classifier )
        summitAIModel.fit ( self.X , self.y )
        joblib.dump ( summitAIModel , dumpfile )

    def load_model( self, modeldump ):
        self.pipe = joblib.load(modeldump)

    def predict( self, pred ):
        return self.pipe.predict(pred)


class SummitAITriage5Areas(SummitAIModel):
    """
    Class for triage in 5 areas: ACCTG, ARCH, BO, FO, RISK
    """
    def __init__( self, dbclient, modeldump, dateFrom, dateTo  ):
        super(SummitAITriage5Areas, self).__init__( dbclient, modeldump, dateFrom, dateTo  )

    def clean_text ( self, text ) :
        """
        Function to clean the text
        :param text: the text for a case
        :return: the cleaned text
        """
        text = BeautifulSoup ( text , "lxml" ).text  # HTML decoding
        text = text.lower ( )  # lowercase text
        text = REPLACE_BY_SPACE_RE.sub ( ' ' , text )  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub ( '' , text )  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = ' '.join ( word for word in text.split ( ) if word not in STOPWORDS )  # delete stopwors from text
        return text

    def loadData( self ):
        dbcases = self.dbclient['newai']['salesforcecases']
        allNewCases = list ( dbcases.find ( {'caseOpenDate' : {'$gte' : self.dateFrom , '$lte' : self.dateTo}} ,
                                            {'caseNumber' : 1 , 'caseSubject' : 1 , 'caseDescription' : 1 ,
                                             'caseModule' : 1} ) )

        allc = []
        for c in allNewCases :
            mod = moduleToArea(c['caseModule'])
            subPlusDesc = c['caseSubject'] + ' ' + c['caseDescription']
            if mod is not None and subPlusDesc is not None:
                self.X.append(subPlusDesc)
                self.y.append( mod )

        self.X = pd.Series(self.X)
        self.y = pd.Series(self.y)


    def buildModel( self, transformer, classifier ):
        self.pipe = make_pipeline ( transformer , classifier )
        self.pipe.fit ( self.X , self.y )
        joblib.dump (self.pipe , self.modeldump)

    def predict( self, pred ):
        return self.pipe.predict(pred)


class SummitAITriageByModule(SummitAIModel):
    """
    Class for triage in 5 areas: ACCTG, ARCH, BO, FO, RISK
    """
    def __init__( self, dbclient, modeldump, dateFrom, dateTo ):
        super(SummitAITriageByModule, self).__init__( dbclient, modeldump, dateFrom, dateTo )
        self.pipes = {}
        self.modelDumps = {
            'acct' : modeldump + "_acct" + MODEL_DUMP_EXT,
            'arch' :  modeldump + "_arch" + MODEL_DUMP_EXT,
            'bo' :  modeldump + "_bo" + MODEL_DUMP_EXT,
            'fo' : modeldump + "_fo" + MODEL_DUMP_EXT,
            'risk' : modeldump + "_risk" + MODEL_DUMP_EXT
        }

        self.X = [[],[],[],[],[]]
        self.y = [[],[],[],[],[]]

        self.ModuleData = {
            'acct'  : 0,
            'arch'  : 1,
            'bo'    : 2,
            'fo'    : 3,
            'risk'  : 4
        }


    def clean_text ( self, text ) :
        """
        Function to clean the text
        :param text: the text for a case
        :return: the cleaned text
        """
        text = BeautifulSoup ( text , "lxml" ).text  # HTML decoding
        text = text.lower ( )  # lowercase text
        text = REPLACE_BY_SPACE_RE.sub ( ' ' , text )  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub ( '' , text )  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = ' '.join ( word for word in text.split ( ) if word not in STOPWORDS )  # delete stopwors from text
        return text

    def loadData( self ):
        dbcases = self.dbclient['newai']['salesforcecases']
        allNewCases = list ( dbcases.find ( {'caseOpenDate' : {'$gte' : self.dateFrom , '$lte' : self.dateTo}} ,
                                            {'caseNumber' : 1 , 'caseSubject' : 1 , 'caseDescription' : 1 ,
                                             'caseModule' : 1} ) )
        allc = []
        for c in allNewCases:
            mod = moduleToArea(c['caseModule'])
            desc = c['caseSubject'] + ' ' + c['caseDescription']

            if mod is not None and desc is not None:
                self.X[self.ModuleData[mod]].append(desc)
                self.y[self.ModuleData[mod]].append( c['caseModule'] )

        for m in ('acct', 'arch', 'bo', 'fo', 'risk'):
            self.X[self.ModuleData[m]] = pd.Series ( self.X[self.ModuleData[m]] )
            self.y[self.ModuleData[m]] = pd.Series ( self.y[self.ModuleData[m]] )
            self.X[self.ModuleData[m]] ,  self.y[self.ModuleData[m]] = occurencesCount (  self.X[self.ModuleData[m]] ,  self.y[self.ModuleData[m]] )



    def buildModel (self , transformer, classifier ):
        """
        Train a model with scikit-learn
        :param X: X-train
        :param y: y_train
        :param transformer: transformer class ( TfidVectorizer )
        :param classifier: classifier class , the model
        :param dumpfile: output -> dump model
        """
        for module in ('acct' , 'arch', 'bo', 'fo', 'risk'):
            summitAIModel = make_pipeline ( transformer , classifier )
            summitAIModel.fit ( self.X[self.ModuleData[module]], self.y[self.ModuleData[module]] )
            joblib.dump ( summitAIModel, self.modelDumps[module] )

    def load_model ( self  ) :
        self.pipes['acct'] = joblib.load ( self.modeldumpAcctg )
        self.pipes['arch'] = joblib.load ( self.modeldumpArch )
        self.pipes['bo'] = joblib.load ( self.modeldumpBo )
        self.pipes['fo'] = joblib.load ( self.modeldumpFo )
        self.pipes['risk'] = joblib.load ( self.modeldumpRisk )

    def predict ( self , pred, module ) :
        return self.pipes[module].predict ( pred )



class SummitAITriage5Areas2(SummitAIModel):
    """
    Class for triage in 5 areas: ACCTG, ARCH, BO, FO, RISK
    """
    def __init__( self, dbclient, modeldump, dateFrom, dateTo  ):
        super(SummitAITriage5Areas2, self).__init__( dbclient, modeldump, dateFrom, dateTo  )

    def clean_text ( self, text ) :
        """
        Function to clean the text
        :param text: the text for a case
        :return: the cleaned text
        """
        text = BeautifulSoup ( text , "lxml" ).text  # HTML decoding
        text = text.lower ( )  # lowercase text
        text = REPLACE_BY_SPACE_RE.sub ( ' ' , text )  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub ( '' , text )  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = ' '.join ( word for word in text.split ( ) if word not in STOPWORDS )  # delete stopwors from text
        return text

    def loadData( self ):
        dbcases = self.dbclient['summitai']['cases']
        allNewCases = list ( dbcases.find ( {'openedDate' : {'$gte' : self.dateFrom , '$lte' : self.dateTo}} ,
                                            {'caseNumber' : 1 , 'subject' : 1 , 'description' : 1 ,
                                             'productModule' : 1} ) )

        allc = []
        for c in allNewCases :
            mod = moduleToArea(c['productModule'])
            subPlusDesc = c['subject'] + ' ' + c['description']
            if mod is not None and subPlusDesc is not None:
                self.X.append(subPlusDesc)
                self.y.append( mod )

        self.X = pd.Series(self.X)
        self.y = pd.Series(self.y)

        self.X, self.y = occurencesCount (  pd.DataFrame({'X': self.X, 'y': self.y}) )


    def buildModel( self, transformer, classifier ):
        summitAIModel = make_pipeline ( transformer , classifier )
        summitAIModel.fit ( self.X , self.y )
        joblib.dump (summitAIModel , self.modeldump)

    def predict( self, pred ):
        return self.pipe.predict(pred)

    def load_model ( self ) :
            self.pipe = joblib.load ( self.modeldump )



    def testModel( self, transformer, classifier  ):
        X_train , X_test , y_train , y_test = train_test_split ( self.X , self.y , test_size = 0.2 , random_state = 42 )
        nb = Pipeline ( [('a' , transformer) ,
                         ('b' , classifier)
                         ] )

        nb.fit ( X_train , y_train )

        from sklearn.metrics import classification_report
        y_pred = nb.predict ( X_test )

        print ( 'accuracy %s' % accuracy_score ( y_pred , y_test ) )
        my_tags = ['acct' , 'arch' , 'bo' , 'fo' , 'risk']
        print ( classification_report ( y_test , y_pred , target_names = my_tags ) )




class SummitAITriageByModule2(SummitAIModel):
    """
    Class for triage in 5 areas: ACCTG, ARCH, BO, FO, RISK
    """
    def __init__( self, dbclient, modeldump, dateFrom, dateTo ):
        super(SummitAITriageByModule2, self).__init__( dbclient, modeldump, dateFrom, dateTo )
        self.pipes = {}
        self.modelDumps = {
            'acct' : modeldump + "_acct" + MODEL_DUMP_EXT,
            'arch' :  modeldump + "_arch" + MODEL_DUMP_EXT,
            'bo' :  modeldump + "_bo" + MODEL_DUMP_EXT,
            'fo' : modeldump + "_fo" + MODEL_DUMP_EXT,
            'risk' : modeldump + "_risk" + MODEL_DUMP_EXT
        }

        self.X = [[],[],[],[],[]]
        self.y = [[],[],[],[],[]]

        self.ModuleData = {
            'acct'  : 0,
            'arch'  : 1,
            'bo'    : 2,
            'fo'    : 3,
            'risk'  : 4
        }


    def clean_text ( self, text ) :
        """
        Function to clean the text
        :param text: the text for a case
        :return: the cleaned text
        """
        text = BeautifulSoup ( text , "lxml" ).text  # HTML decoding
        text = text.lower ( )  # lowercase text
        text = REPLACE_BY_SPACE_RE.sub ( ' ' , text )  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub ( '' , text )  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = ' '.join ( word for word in text.split ( ) if word not in STOPWORDS )  # delete stopwors from text
        return text

    def loadData( self ):
        dbcases = self.dbclient['summitai']['cases']
        allNewCases = list ( dbcases.find ( {'openedDate' : {'$gte' : self.dateFrom , '$lte' : self.dateTo}} ,
                                            {'caseNumber' : 1 , 'subject' : 1 , 'description' : 1 ,
                                             'productModule' : 1} ) )
        allc = []
        for c in allNewCases:
            mod = moduleToArea(c['productModule'])
            desc = c['subject'] + ' ' + c['description']

            if mod is not None and desc is not None:
                self.X[self.ModuleData[mod]].append(desc)
                self.y[self.ModuleData[mod]].append( c['productModule'] )

        for m in ('acct', 'arch', 'bo', 'fo', 'risk'):
            self.X[self.ModuleData[m]] = pd.Series ( self.X[self.ModuleData[m]] )
            self.y[self.ModuleData[m]] = pd.Series ( self.y[self.ModuleData[m]] )
            self.X[self.ModuleData[m]] ,  self.y[self.ModuleData[m]] = occurencesCount (  pd.DataFrame({'X': self.X[self.ModuleData[m]] , 'y': self.y[self.ModuleData[m]]}) )
            self.X[self.ModuleData[m]].apply(self.clean_text)


    def buildModel (self , transformer, classifier ):
        """
        Train a model with scikit-learn
        :param X: X-train
        :param y: y_train
        :param transformer: transformer class ( TfidVectorizer )
        :param classifier: classifier class , the model
        :param dumpfile: output -> dump model
        """
        for module in ('acct' , 'arch', 'bo', 'fo', 'risk'):
            summitAIModel = make_pipeline ( transformer , classifier )
            summitAIModel.fit ( self.X[self.ModuleData[module]], self.y[self.ModuleData[module]] )
            joblib.dump ( summitAIModel, self.modelDumps[module] )

    def load_model ( self  ) :
        self.pipes['acct'] = joblib.load ( self.modelDumps['acct'] )
        self.pipes['arch'] = joblib.load ( self.modelDumps['arch'])
        self.pipes['bo'] = joblib.load ( self.modelDumps['bo'] )
        self.pipes['fo'] = joblib.load ( self.modelDumps['fo'] )
        self.pipes['risk'] = joblib.load ( self.modelDumps['risk'] )

    def predict ( self , pred, module ) :
        return self.pipes[module].predict ( pred )

    def testModels ( self , transformer , classifier, X, y ) :
        X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state = 42 )
        nb = Pipeline ( [ ('a' , transformer) ,
                          ('b' , classifier)
                          ] )

        nb.fit ( X_train , y_train )

        from sklearn.metrics import classification_report
        y_pred = nb.predict ( X_test )

        print ( 'accuracy %s' % accuracy_score ( y_pred , y_test ) )
       # print ( classification_report ( y_test , y_pred , target_names = my_tags ) )

    def testModel( self, transformer, classifier  ):
        for m in ('acct' , 'arch' , 'bo' , 'fo' , 'risk') :
            self.testModels(transformer, classifier,  self.X[self.ModuleData[m]], self.y[self.ModuleData[m]])

class SummitAIAcctg(SummitAIModel):
    """
    Class for triage in 5 areas: ACCTG, ARCH, BO, FO, RISK
    """
    def __init__( self, dbclient, modeldump, dateFrom, dateTo  ):
        super(SummitAIAcctg, self).__init__( dbclient, modeldump, dateFrom, dateTo  )

    def clean_text ( self, text ) :
        """
        Function to clean the text
        :param text: the text for a case
        :return: the cleaned text
        """
        text = BeautifulSoup ( text , "lxml" ).text  # HTML decoding
        text = text.lower ( )  # lowercase text
        text = REPLACE_BY_SPACE_RE.sub ( ' ' , text )  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub ( '' , text )  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = ' '.join ( word for word in text.split ( ) if word not in STOPWORDS )  # delete stopwors from text
        return text

    def loadData( self ):
        dbcases = self.dbclient['summitai']['cases']
        allNewCases = list ( dbcases.find (
            {'openedDate' : {'$gte' : self.dateFrom , '$lte' : self.dateTo} , 'productModule' : {'$in' : ["Summit - Accounting" , "Summit - P&L" ,
              "Summit - Hedge Accounting/FAS133"]} } ,
            {'caseNumber' : 1 , 'subject' : 1 , 'description' : 1 , 'productModule' : 1} ) )

        allc = []
        for c in allNewCases :
            mod =c['productModule']
            subPlusDesc = c['subject'] + ' ' + c['description']
            if mod is not None and subPlusDesc is not None:
                self.X.append(subPlusDesc)
                self.y.append( mod )

        self.X = pd.Series(self.X)
        self.y = pd.Series(self.y)


    def buildModel( self, transformer, classifier ):
        summitAIModel = make_pipeline ( transformer , classifier )
        summitAIModel.fit ( self.X , self.y )
        joblib.dump (summitAIModel , self.modeldump)

    def load_model( self ):
        self.pipe = joblib.load(self.modeldump)

    def predict( self, pred ):
        return self.pipe.predict(pred)



