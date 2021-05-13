
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


df=pd.read_csv(r"C:\Users\דביר\Downloads\voice.csv")

y=[]
for i in range(len(df)):
    if df.iloc[i,-1]=='male':
       y.append(1)
    else:
       y.append(0)
X=df.iloc[:,:-1]   

x_train,x_test,y_train,y_test = train_test_split(X,y)    
#fpr,tpr=[],[]
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
print("logistic regression accuracy is:", logreg.score(x_test,y_test))    

reg = LinearRegression().fit(x_train,y_train)
print("linear regression accuracy is:",reg.score(x_test,y_test))

clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)
print("Decision Tree accuracy is:",clf.score(x_test,y_test))

rand= RandomForestClassifier()
rand.fit(x_train,y_train)
print("Random forest accuracy is:",rand.score(x_test,y_test))
