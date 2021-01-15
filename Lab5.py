import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv("parse_.csv")
df = df.fillna(0)

df['Class'] = df['Name']

df2 = df.drop(labels=['Name', 'City','Company','Descriptions', 'Duties','Requiremenst','Date','Terms'], axis=1)
df2.reset_index(drop=True, inplace=True)
leClass = LabelEncoder()
leClass.fit(list(df2['Class'].astype(str).values))
f = open("Classes.txt", "w")
for i in range(len(leClass.classes_)):
    f.write(str(i)+" "+leClass.classes_[i]+"\n")
f.close()
df2['Class'] = leClass.transform(list(df2['Class'].astype(str).values))

oheExpirience  = OneHotEncoder(sparse=False)
oheExpirience.fit(df2['Expirience'].to_numpy().reshape(-1, 1))
transformed = oheExpirience.transform(df2['Expirience'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheExpirience.get_feature_names())
df2 = pd.concat([df2, ohe_df], axis=1).drop(['Expirience'], axis=1)

oheEmployment  = OneHotEncoder(sparse=False)
oheEmployment.fit(df2['Employment'].to_numpy().reshape(-1, 1))
transformed = oheEmployment.transform(df2['Employment'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheEmployment.get_feature_names())
df2 = pd.concat([df2, ohe_df], axis=1).drop(['Employment'], axis=1)

oheSchedule  = OneHotEncoder(sparse=False)
oheSchedule.fit(df2['WorkSchedule'].to_numpy().reshape(-1, 1))
transformed = oheSchedule.transform(df2['WorkSchedule'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheSchedule.get_feature_names())
df2 = pd.concat([df2, ohe_df], axis=1).drop(['WorkSchedule'], axis=1)
text_transformer = CountVectorizer()
text = text_transformer.fit_transform(df2['KeySkills'].apply(lambda x: np.str_(x)))
words = pd.DataFrame(text.toarray(), columns=text_transformer.get_feature_names())
df2 = pd.concat([df2, words], axis=1).drop(['KeySkills'], axis=1)
data = df2.drop(labels=['Class'], axis=1)
target = df2['Class']
train_data, test_data, train_target, test_target = train_test_split(data,target, test_size=0.3, random_state=0)

# def test_model(model):
#     model.fit(train_data, train_target)
#     print("cv: "+str(model.score(test_data, test_target)))

print("MultinomialNB: 0.3565459610027855")
# model = MultinomialNB() 
# test_model(model)

print("AdaBoostClassifier: 0.35376044568245124")
# model = AdaBoostClassifier(n_estimators=500)
# test_model(model)

print("KNeighborsClassifier: 0.5738161559888579")
# model = KNeighborsClassifier(n_neighbors=20)
# test_model(model)

print("SVC RBF 0.40947075208913647")
# model = SVC(kernel='rbf',gamma='auto')
# test_model(model)

df_new = pd.read_csv("kazan_.csv")
df_new = df_new.fillna(0)
df_new['Class'] = df_new['Name']
copy = df_new.copy()

df_new = df_new.drop(labels=['Name', 'City','Company', 'Descriptions', 'Duties','Requiremenst','Date','Terms'], axis=1)
df_new.reset_index(drop=True, inplace=True)

df_new['Class'] = leClass.transform(list(df_new['Class'].astype(str).values))

transformed = oheExpirience.transform(df_new['Expirience'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheExpirience.get_feature_names())
df_new = pd.concat([df_new, ohe_df], axis=1).drop(['Expirience'], axis=1)

transformed = oheEmployment.transform(df_new['Employment'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheEmployment.get_feature_names())
df_new = pd.concat([df_new, ohe_df], axis=1).drop(['Employment'], axis=1)

transformed = oheSchedule.transform(df_new['WorkSchedule'].to_numpy().reshape(-1, 1))
ohe_df = pd.DataFrame(transformed, columns=oheSchedule.get_feature_names())
df_new = pd.concat([df_new, ohe_df], axis=1).drop(['WorkSchedule'], axis=1)

text = text_transformer.transform(df_new['KeySkills'].apply(lambda x: np.str_(x)))
words = pd.DataFrame(text.toarray(), columns=text_transformer.get_feature_names())
df_new = pd.concat([df_new, words], axis=1).drop(['KeySkills'], axis=1)

data_new = df_new.drop(labels=['Class'], axis=1)
target_new = df_new['Class']
model = KNeighborsClassifier(n_neighbors=20)

model.fit(data, target)
print(str(model.score(data_new, target_new)))
copy["Predicted_class"] = model.predict(data_new)

copy["Predicted_class"] = leClass.inverse_transform(list(copy["Predicted_class"].values))
copy.to_csv("laba5.csv",  na_rep = 'NA', index = True, index_label = "", quotechar = '"', quoting = csv.QUOTE_NONNUMERIC, encoding = "utf-8-sig")
