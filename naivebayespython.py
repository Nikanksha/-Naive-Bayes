from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix 

################## Reading the Salary Data 
salary_train = pd.read_csv("C:\\Users\\Nik\\Downloads\\SalaryData_Train.csv")
salary_test = pd.read_csv("C:\\Users\\Nik\\Downloads\\SalaryData_Test.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 80%

#print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 80%
#Accuracy 0.7946879150066402

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print("Accuracy",(10891+780)/(10891+780+2920+780))  # 75%

#print("Accuracy",(10891+780)/(10891+780+2920+780))  # 75%
#Accuracy 0.7592869689675362
