from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from util import SummitAIDB , SummitAIApp
import schedule
import time
from aiutil import SummitAITriageModel , SummitAITriage5Areas , SummitAITriageByModule , SummitAIAcctg , \
    SummitAITriage5Areas2 , SummitAITriageByModule2
from sklearn.linear_model import SGDClassifier , RidgeClassifier
import pandas as pd
import numpy as np

app = SummitAIApp ( 'config.json' )


#def checkEmails():
#    app.checkCases ( )


print ('Start application....')

transformer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3),
                        min_df=2, max_df=0.9, norm="l2", dtype=np.float32)
classifier =  RidgeClassifier(tol=1e-2, solver="sag")

app.db.testConnection()

app.loadCasesFromCSV('data/triage.csv')

print('data loaded')


'''

print("Create Acctg model")
aiiModelAcctg = SummitAIAcctg(app.db.dbclient, "acctg3", "2010-01-01","2020-06-07")
#aiiModelAcctg.loadData()
#aiiModelAcctg.buildModel( transformer=transformer, classifier=classifier )


print("Model for Acctg created ........")
'''

'''
aiiModelAcctg.load_model()


#app.modifyAcct(aiiModelAcctg)

print("Acctgs modified completed")
'''

'''

print("Creating Modules......")
aiModelAreas = SummitAITriage5Areas2(app.db.dbclient, "summitcvs", "2010-01-01","2020-06-06")
aiModelAreas.loadData()
aiModelAreas.buildModel( transformer=transformer, classifier=classifier )

#aiModelAreas.testModel(transformer=transformer, classifier=classifier)

#aiModelAreas.load_model()
print("Model for Areas loaded........")



aiModelModules = SummitAITriageByModule2(app.db.dbclient, "summitcvs", "2010-01-01","2020-06-06")
aiModelModules.loadData()
aiModelModules.buildModel( transformer=transformer, classifier=classifier )

#aiModelModules.load_model()
#aiModelModules.testModel(transformer=transformer, classifier=classifier)
print("Model for each Modules loaded........")


linkCase = 'https://misys.my.salesforce.com/{}'
'''
'''



def processNewCases():
    newCases = app.printNewCases()

    preds = aiModelAreas.predict(newCases[2])

    triageResults = []
    for i in range(len(newCases)):
        predictedArea = preds[i]
        toPredict = [newCases[2][i]]
        predictedModule = aiModelModules.predict(toPredict, predictedArea)
        triageResults.append({'caseId':  newCases[0][i], 'Subject': newCases[1][i], 'ProductModule': predictedModule[0]})


    print ("=" * 80 )
    print ("PROGRAMM OUTPUT: " )
    for case in triageResults:
        print("*" * 50 )
        print("Case: " + linkCase.format(case['caseId']))
        print("Subject: " + case['Subject'])
        print("Predicted Module: " + case['ProductModule'])
        print()
    print ("=" * 80 )



schedule.every ( 1 ).minutes.do ( processNewCases )

while True :
    # Checks whether a scheduled task
    # is pending to run or not
    schedule.run_pending ( )
    time.sleep ( 1 )
'''