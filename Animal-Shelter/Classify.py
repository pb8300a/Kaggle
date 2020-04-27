# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:24:33 2019

@author: f400563
"""

import pandas as pd
import numpy as np
import os 
import re
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
%matplotlib inline
os.chdir("C:/Users/F400563/Desktop/uci/animal")

trn = pd.read_csv("C:/Users/F400563/Desktop/uci/animal/train.csv",encoding = "ISO-8859-1")
test = pd.read_csv("C:/Users/F400563/Desktop/uci/animal/test.csv",encoding = "ISO-8859-1")

########## #xploratory Analaysis #######
trn.OutcomeType.value_counts()
trn.AnimalType.value_counts()
trn.AnimalType.value_counts()/trn.AnimalType.value_counts().sum()


def dataprep(dataset,traintype):
    
    #dataset = trn
    dataset['is_cat'] = pd.get_dummies(trn.AnimalType).Cat
    
    ### Colors ####
    dataset['is_color_mix'] = dataset.Color.str.contains('/').astype('int64')
    dataset['is_tabby'] = dataset.Color.str.contains('Tabby').astype('int64')
    dataset['is_tick'] = dataset.Color.str.contains('Tick').astype('int64')
    dataset['is_tortie'] = dataset.Color.str.contains('Tortie').astype('int64')
    dataset['is_torbie'] = dataset.Color.str.contains('Torbie').astype('int64')
    dataset['is_point'] = dataset.Color.str.contains('Point').astype('int64')
    
    #how many colors are there per pet
    f = lambda x: ' '.join([item for item in x.split() if item not in ['Tabby','Tick','Tortie','Torbie','Point']])
    new_color = dataset["Color"].apply(f).str.replace('/',' ')
    dataset['nwords_color'] = new_color.str.split().apply(len)
    
    #get the primary color
    dataset['primary_color'] = new_color.str.split().str.get(0)
    
    dataset['has_tan'] = dataset.Color.str.contains('Tan').astype('int64')
    dataset['has_white'] = dataset.Color.str.contains('White').astype('int64')
    dataset['has_black'] = dataset.Color.str.contains('Black').astype('int64')
    dataset['has_orange'] = dataset.Color.str.contains('Orange').astype('int64')
    dataset['has_brown'] = dataset.Color.str.contains('Brown').astype('int64')
    dataset['has_cream'] = dataset.Color.str.contains('Cream').astype('int64')
    dataset['has_tri'] = dataset.Color.str.contains('Tricolor').astype('int64')
    
    #Mixed Breeds
    is_unknown_mix = dataset.Breed.str.contains('Mix').astype('int64')
    is_known_mix = dataset.Breed.str.contains('/').astype('int64')
    dataset['is_mix'] = [0]*dataset.shape[0]
    dataset.loc[(is_unknown_mix == 1) | (is_known_mix == 1),'is_mix'] = 1
    
    #Cat Breeds
    dataset['is_shorthair'] = dataset.Breed.str.contains('Shorthair',flags=re.IGNORECASE).astype('int64') 
    dataset['is_mediumhair'] = dataset.Breed.str.contains('Medium',flags=re.IGNORECASE).astype('int64') 
    dataset['is_longhair'] = dataset.Breed.str.contains('Longhair',flags=re.IGNORECASE).astype('int64') 
    dataset['is_siamese'] = dataset.Breed.str.contains('Siamese',flags=re.IGNORECASE).astype('int64') 
    
    #Dog Breeds
    dataset['is_pit'] = dataset.Breed.str.contains('Pit',flags=re.IGNORECASE).astype('int64') 
    dataset['is_terrier'] = dataset.Breed.str.contains('Terrier',flags=re.IGNORECASE).astype('int64') 
    dataset['is_poodle'] = dataset.Breed.str.contains('Poodle',flags=re.IGNORECASE).astype('int64') 
    dataset['is_lab'] = dataset.Breed.str.contains('Labrador',flags=re.IGNORECASE).astype('int64') 
    dataset['is_schnau'] = dataset.Breed.str.contains('Schnauzer',flags=re.IGNORECASE).astype('int64') 
    dataset['is_collie'] = dataset.Breed.str.contains('Collie',flags=re.IGNORECASE).astype('int64') 
    dataset['is_shepherd'] = dataset.Breed.str.contains('Shepherd',flags=re.IGNORECASE).astype('int64') 
    dataset['is_beagle'] = dataset.Breed.str.contains('Beagle',flags=re.IGNORECASE).astype('int64') 
    dataset['is_chihuahua'] = dataset.Breed.str.contains('Chihuahua',flags=re.IGNORECASE).astype('int64') 
    dataset['is_retriever'] = dataset.Breed.str.contains('Retriever',flags=re.IGNORECASE).astype('int64') 
    dataset['is_miniature'] = dataset.Breed.str.contains('Miniature',flags=re.IGNORECASE).astype('int64') 
    dataset['is_lab'] = dataset.Breed.str.contains('Siamese',flags=re.IGNORECASE).astype('int64') 
    #Big Dogs
    dataset['is_mastiff'] = dataset.Breed.str.contains('Mastiff',flags=re.IGNORECASE).astype('int64') 
    dataset['is_great'] = dataset.Breed.str.contains('Great',flags=re.IGNORECASE).astype('int64') 
    dataset['is_rott'] = dataset.Breed.str.contains('Rott',flags=re.IGNORECASE).astype('int64') 
    dataset['is_hound'] = dataset.Breed.str.contains('Bloodhound',flags=re.IGNORECASE).astype('int64') 
    dataset['is_bernard'] = dataset.Breed.str.contains('Bernard',flags=re.IGNORECASE).astype('int64') 
    dataset['is_akita'] = dataset.Breed.str.contains('Akita',flags=re.IGNORECASE).astype('int64') 
    dataset['is_borzoi'] = dataset.Breed.str.contains('Borzoi',flags=re.IGNORECASE).astype('int64') 
    dataset['is_Beau'] = dataset.Breed.str.contains('Beauceron',flags=re.IGNORECASE).astype('int64') 
    dataset['is_Leon'] = dataset.Breed.str.contains('Leonberger',flags=re.IGNORECASE).astype('int64') 
    
    dataset['is_large_dog'] = [0]*dataset.shape[0]
    
    dataset.loc[((dataset.is_mastiff ==1) | (dataset.is_great ==1) | (dataset.is_rott ==1) | (dataset.is_hound ==1 ) | (dataset.is_bernard==1 ) | (dataset.is_akita ==1 ) | (dataset.is_borzoi==1 ) | (dataset.is_Beau==1 ) | (dataset.is_Leon==1 )),"is_large_dog"] = 1
    
                
    #Gender
    male_names = pd.read_csv("male_names.txt",header = None)
    female_names = pd.read_csv("female_names.txt",header = None)
    names = pd.concat([male_names.iloc[:,0],female_names.iloc[:,0]])
    Sex = ["Male"]*len(male_names)+["Female"]*len(female_names)
    d = {"Sex" : Sex, "Name" : names}
    gender_data = pd.DataFrame(data=d).drop_duplicates(subset = 'Name')
    
    #Name
    dataset['is_named'] = [1]*dataset.shape[0]
    dataset.loc[dataset.Name.isnull(),'is_named']= 0
    dataset = dataset.merge(gender_data,how = 'left', on = 'Name')
    dataset["Gender"] = [2]*dataset.shape[0]
    dataset.loc[dataset.Sex=="Male",'Gender'] =0
    dataset.loc[dataset.Sex=="Female",'Gender'] =1
    
    dataset['not_human_name'] = [0]*dataset.shape[0]
    dataset.loc[(dataset.Gender == 2) & (dataset.is_named == 1),'not_human_name'] =1
               
    dataset['name_ends_with_y'] = dataset.Name.str.endswith('y')
    dataset['name_ends_with_ie'] = dataset.Name.str.endswith('ie')
    dataset.loc[dataset.is_named ==1,'name_ends_with_y'] = False
    dataset.loc[dataset.is_named ==1,'name_ends_with_ie'] = False
    
    dataset['name_ends_with_y_or_ie'] = [0]*dataset.shape[0]
    dataset.loc[(dataset.name_ends_with_y == 1)  | (dataset.name_ends_with_ie ==1),'name_ends_with_y_or_ie'] = 1
    
    ##########################################
    ## Add cheater data and see if it helps ##
    ##########################################
    dataset['gender_true'] = [2]*dataset.shape[0]
    dataset.loc[dataset.SexuponOutcome.str.contains('Male')==True,'gender_true'] = 0
    dataset.loc[dataset.SexuponOutcome.str.contains('Female')==True,'gender_true'] = 1

    is_neutered = dataset.SexuponOutcome.str.contains("Neutered")
    is_spayed = dataset.SexuponOutcome.str.contains("Spayed")
    dataset['is_sterilized'] = [0]*dataset.shape[0]
    dataset.loc[(is_neutered==1) | (is_spayed == 1),"is_sterilized"]=1
    
    dataset.loc[dataset.AgeuponOutcome.isnull(),"AgeuponOutcome"] = "0 years"
    length2 = [item[0] for item in dataset.AgeuponOutcome.str.split()]
    length2 = [float(i) for i in length2]
    
    dataset['unit'] = ['xxx']*dataset.shape[0]
    dataset['unit_data'] =  [item[1] for item in dataset.AgeuponOutcome.str.split()]
    dataset.loc[dataset.unit_data.isin(['week','weeks'])==True,'unit' ] = 7
    dataset.loc[dataset.unit_data.isin(['day','days'])==True,'unit' ] = 1
    dataset.loc[dataset.unit_data.isin(['month','months'])==True,'unit' ] = 30
    dataset.loc[dataset.unit_data.isin(['year','years'])==True,'unit' ] = 365
    
    dataset['age'] = [0]*dataset.shape[0]
    dataset['age'] = dataset.unit*length2/365

    dataset.age_unknown = dataset.age.astype('bool')
    dataset.DateTime = pd.Series(dataset.DateTime ).str.replace('-','/')
    if traintype == "test":
        dataset.DateTime = dataset.DateTime.str[:-3]
    
    date = [item[0] for item in dataset.DateTime.str.split()]
    dataset['time_of_day'] = [item[1] for item in dataset.DateTime.str.split()]
    hour_of_day = [item[0] for item in dataset.time_of_day.str.split(":")]
    dataset['hour_of_day'] = [float(i) for i in hour_of_day]

    dataset['month'] = [item[0] for item in pd.Series(date).str.split("/")]
    dataset.month = dataset.month.astype('int64')
    dataset['year'] = [item[2] for item in pd.Series(date).str.split("/")]
    dataset.year = dataset.year.astype('int64')

    if traintype == "test":
        day_of_week=pd.to_datetime(dataset.DateTime, format = '%Y/%m/%d %H:%M').dt.weekday_name
    else:
        day_of_week=pd.to_datetime(dataset.DateTime, format = '%m/%d/%Y %H:%M').dt.weekday_name
        dataset["y"] = dataset.OutcomeType.astype('category')
               
    dummy_day_of_week = pd.get_dummies(day_of_week)
    dataset = pd.concat([dataset,dummy_day_of_week],axis = 1)
    
    if traintype == "trn":
        traindata = dataset.loc[:,["is_color_mix","nwords_color","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","month","year","hour_of_day","is_named","is_cat","is_sterilized","age","gender_true","not_human_name","name_ends_with_y_or_ie","is_tabby","is_mix","is_large_dog","is_chihuahua","is_terrier","is_poodle","is_lab","is_schnau","is_retriever","is_pit","is_miniature","is_collie","is_shepherd","is_beagle","is_shorthair","is_mediumhair","is_siamese","is_longhair","is_tortie","is_torbie","is_point","is_tick","y"]]
    else:
        traindata = dataset.loc[:,["is_color_mix","nwords_color","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","month","year","hour_of_day","is_named","is_cat","is_sterilized","age","gender_true","not_human_name","name_ends_with_y_or_ie","is_tabby","is_mix","is_large_dog","is_chihuahua","is_terrier","is_poodle","is_lab","is_schnau","is_retriever","is_pit","is_miniature","is_collie","is_shepherd","is_beagle","is_shorthair","is_mediumhair","is_siamese","is_longhair","is_tortie","is_torbie","is_point","is_tick"]]
    return traindata

trn_data = dataprep(trn,"trn")
test_data = dataprep(test,"test")

Y_trn = trn_data["y"]
trn_data=trn_data.drop('y',axis = 1)
X_train,X_hld,y_train,y_hld = train_test_split(trn_data,Y_trn,test_size = 0.2,random_state = 123123)
watchlist = [ (X_train,'train'), (X_hld, 'test') ]

##################
#Random Forest
#################

rf_model = RandomForestClassifier()

folds = 5
param_comb = 72
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

params_rf_cv = {
          'max_depth' : [5,7,9,11,13,15],
          'min_samples_split' : [4,6],
          #'max_leaf_nodes' : None,
          'n_estimators' :[150,250],
          'min_samples_leaf' : [2,4,6]
 }
random_search_rf = RandomizedSearchCV(rf_model,
                                   param_distributions=params_rf_cv,
                                   n_iter=param_comb,
                                   scoring='neg_log_loss',
                                   n_jobs=2,
                                   cv=skf.split(trn_data,Y_trn),
                                   verbose=3,
                                   random_state=1001 )


random_search_rf.fit(trn_data, Y_trn)

#look at how the scores varied by the hyperparameters for fun
plt.plot(random_search_rf.cv_results_['split0_test_score'])
plt.plot(random_search_rf.cv_results_['split1_test_score'])
plt.plot(random_search_rf.cv_results_['split2_test_score'])
plt.plot(random_search_rf.cv_results_['split3_test_score'])
plt.plot(random_search_rf.cv_results_['split4_test_score'])

#Hyperparameter plots
plt.plot(random_search_rf.cv_results_['param_max_depth'].data)
plt.plot(random_search_rf.cv_results_['param_min_samples_leaf'].data)
plt.plot(random_search_rf.cv_results_['param_min_samples_split'].data)
plt.plot(random_search_rf.cv_results_['param_n_estimators'].data)

print('\n Best estimator: %s' % (random_search_rf.best_estimator_))
print('\n Best Score: %s' % (random_search_rf.best_score_))

new_params_rf = random_search_rf.best_params_
rf_model_final = RandomForestClassifier( criterion= 'gini',
                                         max_depth= new_params_rf['max_depth'],
                                         #max_leaf_nodes= new_params_rf['max_leaf_nodes'],
                                         min_samples_split= new_params_rf['min_samples_split'],
                                         n_estimators = new_params_rf['n_estimators']
                                        )
rf_model_final.fit(X_train,y_train)
rf_model_final.predict(X_hld) 
rf_model_final.predict_proba(X_hld) 
hld_loss = log_loss(y_hld,rf_model_final.predict_proba(X_hld) )
print(hld_loss)



#########
#XGBOOST
########
params_search = {'min_child_weight' : [2,4],
          'max_depth' : [5,6,7,8],
          'learning_rate' : [0.005,0.01,0.015],
          'colsample_bylevel': [0.5,0.75,1], 
          'colsample_bytree': [0.5,0.75,1],
          'objective' : 'multi:softprob'
 }


xgb_model = XGBClassifier(silent = False,n_estimators = 750)

folds = 5
param_comb = 50
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
random_search_xgb = RandomizedSearchCV(xgb_model,
                                   param_distributions=params_search,
                                   n_iter=param_comb,
                                   scoring='neg_log_loss',
                                   n_jobs=4,
                                   cv=skf.split(trn_data,Y_trn),
                                   verbose=3,
                                   random_state=1001 )

random_search_xgb.fit(trn_data, Y_trn)

print('\n Best estimator: %s' % (random_search_xgb.best_estimator_))
print('\n Best Score: %s' % (random_search_xgb.best_score_))
#### Create a final model using CV Results ####
new_params = random_search_xgb.best_params_
new_params['objective'] = 'multi:softprob'


final_xgb_model = XGBClassifier(colsample_bylevel= new_params['colsample_bylevel'],
                                colsample_bytree= new_params['colsample_bytree'],
                                learning_rate= new_params['learning_rate'],
                                max_depth= new_params['max_depth'],
                                min_child_weight= new_params['min_child_weight'],
                                objective= new_params['objective'],
                                n_estimators = 1000,
                                silent = False)
final_xgb_model.fit(X_train, y_train,early_stopping_rounds = 25, eval_metric="mlogloss", eval_set=[(X_hld, y_hld)], verbose=25)


#model.fit(X_train,y_train)
#predictions = model.predict(X_hld)
#from sklearn.metrics import confusion_matrix
#confusion_matrix(predictions, y_hld)
