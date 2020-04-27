##############################################
## Test out Kaggle Shelter Animal Project
## We are not going to play by the rules, instead
## we will just see if we can get good prediction.
##############################################
set.seed(123123)
setwd("C:/Users/F400563/Desktop/uci/animal")
library(tm)
library(ranger)
library(caret)
library(xgboost)
#library(mxnet)

#Load the data
trn = read.csv("train.csv")
test = read.csv("test.csv")


###############################
### Exploratory Analysis ######
###############################
table(trn$OutcomeType)
table(trn$AnimalType)
table(trn$AnimalType)/sum(table(trn$AnimalType))

### There are lots of breeds
nbreeds = length(table(trn$Breed))

#######################
### Dataprep ######
#######################
dataprep = function(dataset,traintest){
        
        #dataset = test
        #### Cat or Dog #####
        dataset$is_cat = ifelse(dataset$AnimalType=="Cat",1,0)
        
        #### Colors #####
        dataset$is_color_mix = ifelse(grepl("\\/",dataset$Color)==TRUE,1,0)
        dataset$is_tabby = ifelse(grepl("Tabby",dataset$Color)==TRUE,1,0)
        dataset$is_tick = ifelse(grepl("Tick",dataset$Color)==TRUE,1,0)
        dataset$is_tortie = ifelse(grepl("Tortie",dataset$Color)==TRUE,1,0)
        dataset$is_torbie = ifelse(grepl("Torbie",dataset$Color)==TRUE,1,0)
        dataset$is_tor = dataset$is_tortie+dataset$is_torbie
        dataset$is_point = ifelse(grepl("Point",dataset$Color)==TRUE,1,0)
        
        #Remove extra words that are not color-related, and see how many colors are actually identified
        dataset$new_color = removeWords(as.character(dataset$Color),c("Tabby","Tick","Tortie","Torbie","Point"))
        dataset$new_color = trimws(gsub("/"," ",dataset$new_color))##############################################
## Test out Kaggle Shelter Animal Project
## We are not going to play by the rules, instead
## we will just see if we can get good prediction.
##############################################
set.seed(123123)
setwd("C:/Users/F400563/Desktop/uci/animal")
library(tm)
library(ranger)
library(caret)
library(xgboost)
#library(mxnet)

#Load the data
trn = read.csv("train.csv")
test = read.csv("test.csv")


###############################
### Exploratory Analysis ######
###############################
table(trn$OutcomeType)
table(trn$AnimalType)
table(trn$AnimalType)/sum(table(trn$AnimalType))

### There are lots of breeds
nbreeds = length(table(trn$Breed))

#######################
### Dataprep ######
#######################
dataprep = function(dataset,traintest){
        
        #dataset = trn
        #### Cat or Dog #####
        dataset$is_cat = ifelse(dataset$AnimalType=="Cat",1,0)
        
        #### Colors #####
        dataset$is_color_mix = ifelse(grepl("\\/",dataset$Color)==TRUE,1,0)
        dataset$is_tabby = ifelse(grepl("Tabby",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_tick = ifelse(grepl("Tick",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_tortie = ifelse(grepl("Tortie",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_torbie = ifelse(grepl("Torbie",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_tor = dataset$is_tortie+dataset$is_torbie
        dataset$is_point = ifelse(grepl("Point",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        
        #Remove extra words that are not color-related, and see how many colors are actually identified
        dataset$new_color = removeWords(as.character(dataset$Color),c("Tabby","Tick","Tortie","Torbie","Point"))
        dataset$new_color = trimws(gsub("/"," ",dataset$new_color))
        dataset$new_color = trimws(gsub("  "," ",dataset$new_color))
        dataset$nwords_color = sapply(strsplit(dataset$new_color, " "), length)
        
        #Get the first color/primary color
        one_word = dataset$new_color[dataset$nwords_color==1]
        two_word = dataset$new_color[dataset$nwords_color==2]
        two_word_first=unlist(lapply(strsplit(two_word," "), function(x) x[1]))
        three_word = dataset$new_color[dataset$nwords_color==3]
        three_word_first=unlist(lapply(strsplit(three_word," "), function(x) x[1]))
        four_word = dataset$new_color[dataset$nwords_color==4]
        four_word_first=unlist(lapply(strsplit(four_word," "), function(x) x[1]))
        
        #Start with just the primary color, combine when necessary
        dataset$primary_color = rep("",nrow(dataset))
        dataset$primary_color[dataset$nwords_color==1] = one_word
        dataset$primary_color[dataset$nwords_color==2] = two_word_first
        dataset$primary_color[dataset$nwords_color==3] = three_word_first
        dataset$primary_color[dataset$nwords_color==4] = four_word_first
        
        dataset$has_tan = ifelse(grepl("Tan",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_white = ifelse(grepl("White",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_black = ifelse(grepl("Black",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_orange = ifelse(grepl("Orange",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_brown = ifelse(grepl("Brown",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_cream = ifelse(grepl("Cream",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_tri = ifelse(grepl("Tricolor",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        
        #### Mixed Breeds #####
        dataset$is_unknown_mix = ifelse(grepl("Mix",dataset$Breed)==TRUE,1,0)
        dataset$is_known_mix = ifelse(grepl("\\/",dataset$Breed)==TRUE,1,0)
        dataset$is_mix = (dataset$is_unknown_mix==1 | dataset$is_known_mix == 1)
        
        ###Cat Breeds####
        dataset$is_shorthair = ifelse(grepl("Shorthair",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_mediumhair = ifelse(grepl("Medium",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_longhair = ifelse(grepl("Longhair",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_siamese = ifelse(grepl("Siamese",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        
        ###Dog Breeds####
        dataset$is_pit = ifelse(grepl("Pit",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_terrier = ifelse(grepl("Terrier",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_poodle = ifelse(grepl("Poodle",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_lab = ifelse(grepl("Labrador",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_schnau = ifelse(grepl("Schnauzer",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_collie = ifelse(grepl("Collie",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_shepherd = ifelse(grepl("Shepherd",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_beagle = ifelse(grepl("Beagle",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_chihuahua = ifelse(grepl("Chihuahua",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_retriever = ifelse(grepl("Retriever",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_miniature = ifelse(grepl("Miniature",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        
        #Big Dogs have more trouble?
        dataset$is_mastiff = ifelse(grepl("Mastiff",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_great = ifelse(grepl("Great",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_rott = ifelse(grepl("Rott",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_hound = ifelse(grepl("Bloodhound",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_bernard = ifelse(grepl("Bernard",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_akita = ifelse(grepl("Akita",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_borzoi = ifelse(grepl("Borzoi",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_Beau = ifelse(grepl("Beauceron",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_Leon = ifelse(grepl("Leonberger",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        
        dataset$is_large_dog = c(dataset$is_mastiff ==1 | dataset$is_great ==1 | dataset$is_rott ==1 | 
                                         dataset$is_hound == 1 | dataset$is_bernard==1 | dataset$is_akita ==1 |
                                         dataset$is_borzoi==1 |  dataset$is_Beau==1 | dataset$is_Leon==1 )
        
        
        
        #### Gender #####
        male_names = as.character(read.table("male_names.txt")[,1])
        female_names = as.character(read.table("female_names.txt")[,1])
        Name = c(male_names,female_names)
        external_gender = c(rep(1,length(male_names)),rep(0,length(female_names)))
        gender_data = cbind.data.frame(Name,external_gender)
        gender_data=gender_data[!duplicated(gender_data$Name),]
        
        #### Name ####
        dataset$is_named = ifelse(dataset$Name=="",0,1)
        if(traintest == "trn"){
                dataset$y = as.factor(dataset$OutcomeType)
                dataset=dataset[order(dataset$AnimalID),]
        }else{
                dataset=dataset[order(dataset$ID),]
        }
        
        dataset = merge(dataset,gender_data,all.x = TRUE,by.x = "Name")
        dataset[!(dataset$external_gender %in% c(1,0)),"external_gender"]= 2
        if(traintest == "trn"){
                dataset=dataset[order(dataset$AnimalID),]
        }else{
                dataset=dataset[order(dataset$ID),]
        }
        
        dataset$not_human_name = rep(0,length(dataset$is_named))
        dataset$not_human_name[dataset$external_gender==2 & dataset$Name != ""] = 1
        dataset$name_ends_withy = endsWith(as.character(dataset$Name), "y")
        dataset$name_ends_withie = endsWith(as.character(dataset$Name), "ie")
        dataset$name_ends_with_y_or_ie = ifelse(dataset$name_ends_withy==1 | dataset$name_ends_withie ==1,1,0)
        
        ##########################################
        ## Add cheater data and see if it helps ##
        ##########################################
        dataset$sex_true = ifelse(grepl("Male",dataset$SexuponOutcome)==TRUE,2,ifelse(grepl("Female",dataset$SexuponOutcome)==TRUE,1,0))
        
        dataset$is_neutered = ifelse(grepl("Neutered",dataset$SexuponOutcome)==TRUE,1,0)
        dataset$is_spayed = ifelse(grepl("Spayed",dataset$SexuponOutcome)==TRUE,1,0)
        dataset$is_sterilized = ifelse(dataset$is_neutered ==1 | dataset$is_spayed == 1,1,0)
        
        dataset$length=as.numeric(unlist(lapply(strsplit(as.character(dataset$AgeuponOutcome)," "), function(x) x[1])))
        dataset$unit=unlist(lapply(strsplit(as.character(dataset$AgeuponOutcome)," "), function(x) x[2]))
        dataset$unit2=rep(0,length(dataset$unit))
        dataset$unit2[dataset$unit %in% c("week","weeks")] = 7
        dataset$unit2[dataset$unit %in% c("day","days")] = 1
        dataset$unit2[dataset$unit %in% c("month","months")] = 30
        dataset$unit2[dataset$unit %in% c("year","years")] = 365
        dataset$age = dataset$length*dataset$unit2/365
        dataset$age[is.na(dataset$age)] = 0
        dataset$age_unknown = (dataset$age == 0)
        dataset$length = NULL;dataset$unit = NULL
        
        dataset$date=unlist(lapply(strsplit(as.character(dataset$DateTime)," "), function(x) x[1]))
        dataset$time_of_day=unlist(lapply(strsplit(as.character(dataset$DateTime)," "), function(x) x[2]))
        dataset$hour_of_day=as.numeric(as.character(unlist(lapply(strsplit(dataset$time_of_day,":"), function(x) x[1]))))
        
        if(traintest == "trn"){
                dataset$month=as.numeric(unlist(lapply(strsplit(as.character(dataset$date),"/"), function(x) x[1])))
                dataset$year=as.numeric(unlist(lapply(strsplit(as.character(dataset$date),"/"), function(x) x[3])))
                dataset$day=weekdays(as.Date(dataset$date,format = "%m/%d/%Y"))
                weekend = (dataset$day %in% c("Friday","Saturday","Sunday"))
                
        }else if(traintest == "test"){
                dataset$month=as.numeric(unlist(lapply(strsplit(as.character(dataset$date),"-"), function(x) x[2])))
                dataset$year=as.numeric(unlist(lapply(strsplit(as.character(dataset$date),"-"), function(x) x[1])))  
                dataset$day=weekdays(as.Date(dataset$date,format = "%Y-%m-%d"))
                weekend = (dataset$day %in% c("Friday","Saturday","Sunday"))
                
        }
        dataset$date = NULL
        
        dmy <- dummyVars(~day, data = dataset,fullRank = FALSE)
        dataset_dummy <- data.frame(predict(dmy, newdata = dataset))
        dataset = cbind.data.frame(dataset,dataset_dummy)
        
        if(traintest =="trn"){
                traindata = dataset[,c("is_color_mix","nwords_color",
                                       "dayMonday","dayTuesday","dayWednesday","dayThursday","dayFriday","daySaturday","daySunday",
                                       "month","year","hour_of_day",
                                       "is_named","is_cat","is_sterilized","age","sex_true",
                                       "not_human_name","name_ends_with_y_or_ie",
                                       "is_tabby","is_mix","is_large_dog","is_chihuahua",
                                       "is_terrier","is_poodle","is_lab","is_schnau","is_retriever","is_pit",
                                       "is_miniature","is_collie","is_shepherd","is_beagle",
                                       "is_shorthair","is_mediumhair","is_siamese","is_longhair","is_tortie","is_torbie","is_point","is_tick",
                                       "y")]
                
        }else if(traintest == "test"){
                traindata = dataset[,c("is_color_mix","nwords_color",
                                       "dayMonday","dayTuesday","dayWednesday","dayThursday","dayFriday","daySaturday","daySunday",
                                       "month","year","hour_of_day",
                                       "is_named","is_cat","is_sterilized","age","sex_true",
                                       "not_human_name","name_ends_with_y_or_ie",
                                       "is_tabby","is_mix","is_large_dog","is_chihuahua",
                                       "is_terrier","is_poodle","is_lab","is_schnau","is_retriever","is_pit",
                                       "is_miniature","is_collie","is_shepherd","is_beagle",
                                       "is_shorthair","is_mediumhair","is_siamese","is_longhair","is_tortie","is_torbie","is_point","is_tick"
                )]
                
        }
        return(traindata)
}



trn_data = dataprep(trn,"trn")
test_data = dataprep(test,"test")
traindat = createDataPartition(trn_data$y, p = 0.8,list = FALSE)
trn_pp = trn_data[traindat,]
hld_pp = trn_data[-traindat,]
yindex = which(names(trn_pp)=="y")
new_y_trn = as.numeric(as.factor(trn_pp$y))-1
new_y_hld = as.numeric(as.factor(hld_pp$y))-1
#######################
#### Random Forest ####
#######################
tControl = trainControl(
        method = "cv",
        number = 5,
        classProbs = TRUE,
        summaryFunction = mnLogLoss,
        verboseIter = T)


tGrid <- expand.grid(  mtry = c(5,7,9,13),
                       splitrule = c("gini"),
                       min.node.size = c(2,4,6))

model_rf = train(y~.,
                       data = trn_pp,
                       trControl = tControl,
                       tuneGrid = tGrid,
                       method = "ranger",
                       metric = "logLoss",
                       importance = "impurity",
                       num.trees = 150) #change to 250

model_rf_pred = predict(model_rf, hld_pp[,-which(names(hld_pp)=="y")])
model_rf_pred2=rep(0,length(model_rf_pred))
model_rf_pred2[model_rf_pred=="Adoption"] = 0
model_rf_pred2[model_rf_pred=="Died"] = 1
model_rf_pred2[model_rf_pred=="Euthanasia"] = 2
model_rf_pred2[model_rf_pred=="Return_to_owner"] = 3
model_rf_pred2[model_rf_pred=="Transfer"] = 4
levels(model_rf_pred2)= list("Adoption"="0","Died"="1","Euthanasia"="2","Return_to_owner"="3","Transfer"="4")

confuse_hld_rf = confusionMatrix(factor(model_rf_pred2),
                              factor(new_y_hld),
                              mode = "everything")


##################
#### Xgboost #####
##################
#### Cross Validation to find best Parameters ####

nround_cand = c(2500) 
eta_cand = c(0.005,0.01,0.015)

max_depth_cand=  c(5,7,9) #deeper trees (10,15) lead to overfit
colsample_bytree_cand = c(0.75)
subsample_cand = c(0.75)

best_minlogloss=100
nround_best=5
dtrn <- xgb.DMatrix(label = new_y_trn, data = as.matrix(trn_pp[,-yindex]))
dhld <- xgb.DMatrix(label = new_y_hld, data = as.matrix(hld_pp[,-yindex]))
watch <- list(train=dtrn, test = dhld)

for(i in 1:length(eta_cand)){
        for(j in 1:length(max_depth_cand)){
                xgb_cv_params <- list(objective = "multi:softprob",
                                   eval_metric = "mlogloss",
                                   num_class = 5,
                                   max_depth = max_depth_cand[j],
                                   subsample = subsample_cand,
                                   seed = 2,
                                   eta = eta_cand[i],
                                   colsample_bytree = colsample_bytree_cand ,
                                   min_child_weight =4)
                
                xgb_cv <- xgb.cv(data = dtrn, 
                               params = xgb_cv_params,
                               nround = nround_cand,
                               verbose =1,
                               print_every_n=100,
                               nfold = 5,
                               early_stopping_rounds = 50,
                               watch)
                
                plot(xgb_cv$evaluation_log[[2]],type = "l",
                     ylim = c(c(0.5,1.75)),
                     main = paste("max_depth_cand: ",max_depth_cand[j],
                                  "eta_cand: ", eta_cand[i]))
                lines(xgb_cv$evaluation_log[[4]],type = "l")
                text(x=200,y = 1.4,min(xgb_cv$evaluation_log[[4]]))
                
                minlogloss = min(xgb_cv$evaluation_log[[4]])
                minlogloss_iter = which(xgb_cv$evaluation_log[[4]] == minlogloss)
                
                if(minlogloss < best_minlogloss ){
                        best_minlogloss = minlogloss
                        xgb_best_params = xgb_cv_params
                        nround_best = minlogloss_iter
                }
        
        }

}
print(xgb_best_params)        
#### Run final model ####
xgb_final <- xgb.train(data = dtrn, 
                 params = xgb_best_params,
                 nround = nround_best+300,
                 verbose =1,
                 print_every_n=40,
                 watch)  
#Hld Predictions
y_pred <- predict(xgb_final, data.matrix(hld_pp[,-yindex]))
hld_preds = data.frame(matrix(y_pred, ncol = 5,byrow = TRUE))
hld_preds$rowmax <- apply(hld_preds, 1, FUN=max)
var = rep(7,nrow(hld_preds))
var[hld_preds$rowmax==hld_preds$X1] = 0
var[hld_preds$rowmax==hld_preds$X2] = 1
var[hld_preds$rowmax==hld_preds$X3] = 2
var[hld_preds$rowmax==hld_preds$X4] = 3
var[hld_preds$rowmax==hld_preds$X5] = 4

confuse_hld_xgb = confusionMatrix(factor(var),
                factor(new_y_hld),
                mode = "everything")

importance = xgb.importance(colnames(dtrn), model = xgb_final)

print(importance)


#### Test Prediction #####
#test_pred <- predict(xgb_final, data.matrix(test_data))
#test_pred2 = data.frame(matrix(test_pred, ncol = 5,byrow = TRUE))

#test_pred2 = cbind.data.frame(test$ID,test_pred2)
#names(test_pred2) = c("ID","Adoption","Died","Euthanasia","Return_to_owner","Transfer")
#write.csv(test_pred2,"submission2.csv")


###################################
#### Neural net using mxnet   #####
###################################
TControl <- trainControl(method = 'cv',
                         number = 5,
                         classProbs = TRUE,
                         verboseIter = FALSE,
                         summaryFunction = mnLogLoss)
tGrid = expand.grid(layer1 = c(30), #(30,0,0) & learningrate = 0.0005 & num.round = 50 seems OK
                    layer2 = c(0),
                    layer3 = c(0),
                    beta1 = c(0.95),
                    beta2 =c(0.95),
                    learningrate = c(0.001),
                    dropout = c(0.1), #0.1
                    activation = "relu")
nn_fit2 <- train(y~.,
              data = trn_pp,
              method = 'mxnetAdam',
              preProcess = c('center', 'scale'),
              trControl = TControl,
              num.round = 50,
              tuneGrid=tGrid,
              na.action = na.exclude)

nn_fit_prob = predict(nn_fit2, hld_pp[,-which(names(hld_pp)=="y")],type = "prob")
nn_fit_class = predict(nn_fit2, hld_pp[,-which(names(hld_pp)=="y")])

nn_fit_class2=rep(0,length(nn_fit_class))
nn_fit_class2[nn_fit_class=="Adoption"] = "0"
nn_fit_class2[nn_fit_class=="Died"] = "1"
nn_fit_class2[nn_fit_class=="Euthanasia"] = "2"
nn_fit_class2[nn_fit_class=="Return_to_owner"] = "3"
nn_fit_class2[nn_fit_class=="Transfer"] ="4"
nn_fit_class2 = as.factor(nn_fit_class2)

confuse_hld_nn = confusionMatrix(factor(nn_fit_class2),
                                 factor(new_y_hld),
                                 mode = "everything")


########################
##### Toy Ensemble #####
########################
rf_pred = predict(model_rf, hld_pp[,-which(names(hld_pp)=="y")],type = "prob")
xgb_pred = data.frame(matrix(predict(xgb_final, data.matrix(hld_pp[,-yindex])), ncol = 5,byrow = TRUE))
nn_pred = predict(nn_fit2, hld_pp[,-which(names(hld_pp)=="y")],type = "prob")

weights = expand.grid(rf_wgt=seq(0,1,.01),xgb_wgt=seq(0,1,.01),nn_wgt=seq(0,1,.01))
weights = weights[rowSums(weights)==1,]
Accuracy= NULL
for(j in 1:nrow(weights)){
        ensemble_pred = rf_pred;for(i in 1:5){ensemble_pred[,i]=rep(0,nrow(ensemble_pred))}
        for(i in 1:5){
                ensemble_pred[,i] = (weights[j,1]*rf_pred[,i]+weights[j,2]*xgb_pred[,i]+weights[j,3]*nn_pred[,i])
        }
        ensemble_pred$rowmax <- apply(ensemble_pred, 1, FUN=max)
        var_ensemble = rep(7,nrow(ensemble_pred))
        var_ensemble[abs(ensemble_pred$rowmax-ensemble_pred$Adoption)<0.01] = 0
        var_ensemble[abs(ensemble_pred$rowmax-ensemble_pred$Died)<0.01] = 1
        var_ensemble[abs(ensemble_pred$rowmax-ensemble_pred$Euthanasia)<0.01] = 2
        var_ensemble[abs(ensemble_pred$rowmax-ensemble_pred$Return_to_owner)<0.01] = 3
        var_ensemble[abs(ensemble_pred$rowmax-ensemble_pred$Transfer)<0.01] = 4
        confuse_hld_ensemble = confusionMatrix(factor(var_ensemble),
                                      factor(new_y_hld),
                                      mode = "everything")
        Accuracy = c(Accuracy,confuse_hld_ensemble$overall[1])

        
        
        #print(paste("Accuracy:", confuse_hld_ensemble$overall[1], "/ W1:",weights[j,1],"  W2:",weights[j,2],"  W3:",weights[j,3]),sep = "")
}

final=cbind.data.frame(Accuracy,weights)
final = final[order(final$Accuracy),]


#### Ensemble Kaggle Prediction #####
xgb_test_pred <- predict(xgb_final, data.matrix(test_data),type = "prob")
xgb_test_pred2 = data.frame(matrix(xgb_test_pred, ncol = 5,byrow = TRUE))

rf_test_pred <- predict(model_rf,newdata = test_data, type = "prob")
nn_test_pred <- predict(nn_fit2,newdata = test_data,type = "prob")

ensemble_test_pred = rf_test_pred;for(i in 1:5){ensemble_test_pred[,i]=rep(0,nrow(ensemble_test_pred))}
for(i in 1:5){
        ensemble_test_pred[,i] = (mean(final[c((nrow(final)-25):nrow(final)),"rf_wgt"])*rf_test_pred[,i]+
                                  mean(final[c((nrow(final)-25):nrow(final)),"xgb_wgt"])*xgb_test_pred2[,i]+
                                  mean(final[c((nrow(final)-25):nrow(final)),"nn_wgt"])*nn_test_pred[,i])
}

final_pred = cbind.data.frame(test$ID,ensemble_test_pred)
names(final_pred) = c("ID","Adoption","Died","Euthanasia","Return_to_owner","Transfer")
write.csv(final_pred,"submission3.csv")


        dataset$new_color = trimws(gsub("  "," ",dataset$new_color))
        dataset$nwords_color = sapply(strsplit(dataset$new_color, " "), length)
        
        #Get the first color/primary color
        one_word = dataset$new_color[dataset$nwords_color==1]
        two_word = dataset$new_color[dataset$nwords_color==2]
        two_word_first=unlist(lapply(strsplit(two_word," "), function(x) x[1]))
        three_word = dataset$new_color[dataset$nwords_color==3]
        three_word_first=unlist(lapply(strsplit(three_word," "), function(x) x[1]))
        four_word = dataset$new_color[dataset$nwords_color==4]
        four_word_first=unlist(lapply(strsplit(four_word," "), function(x) x[1]))
        
        #Start with just the primary color, combine when necessary
        dataset$primary_color = rep("",nrow(dataset))
        dataset$primary_color[dataset$nwords_color==1] = one_word
        dataset$primary_color[dataset$nwords_color==2] = two_word_first
        dataset$primary_color[dataset$nwords_color==3] = three_word_first
        dataset$primary_color[dataset$nwords_color==4] = four_word_first
        
        dataset$has_tan = ifelse(grepl("Tan",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_white = ifelse(grepl("White",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_black = ifelse(grepl("Black",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_orange = ifelse(grepl("Orange",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_brown = ifelse(grepl("Brown",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_cream = ifelse(grepl("Cream",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        dataset$has_tri = ifelse(grepl("Tricolor",dataset$Color,ignore.case = TRUE)==TRUE,1,0)
        
        #### Mixed Breeds #####
        dataset$is_unknown_mix = ifelse(grepl("Mix",dataset$Breed)==TRUE,1,0)
        dataset$is_known_mix = ifelse(grepl("\\/",dataset$Breed)==TRUE,1,0)
        dataset$is_mix = (dataset$is_unknown_mix==1 | dataset$is_known_mix == 1)
        
        ###Cat Breeds####
        dataset$is_shorthair = ifelse(grepl("Shorthair",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_mediumhair = ifelse(grepl("Medium",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_longhair = ifelse(grepl("Longhair",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_siamese = ifelse(grepl("Siamese",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        
        ###Dog Breeds####
        dataset$is_pit = ifelse(grepl("Pit",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_terrier = ifelse(grepl("Terrier",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_poodle = ifelse(grepl("Poodle",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_lab = ifelse(grepl("Labrador",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_schnau = ifelse(grepl("Schnauzer",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_collie = ifelse(grepl("Collie",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_shepherd = ifelse(grepl("Shepherd",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_beagle = ifelse(grepl("Beagle",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_chihuahua = ifelse(grepl("Chihuahua",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_retriever = ifelse(grepl("Retriever",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_miniature = ifelse(grepl("Miniature",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        
        #Big Dogs have more trouble?
        dataset$is_mastiff = ifelse(grepl("Mastiff",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_great = ifelse(grepl("Great",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_rott = ifelse(grepl("Rott",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_hound = ifelse(grepl("Bloodhound",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_bernard = ifelse(grepl("Bernard",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_akita = ifelse(grepl("Akita",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_borzoi = ifelse(grepl("Borzoi",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_Beau = ifelse(grepl("Beauceron",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        dataset$is_Leon = ifelse(grepl("Leonberger",dataset$Breed,ignore.case = TRUE)==TRUE,1,0)
        
        dataset$is_large_dog = c(dataset$is_mastiff ==1 | dataset$is_great ==1 | dataset$is_rott ==1 | 
                                         dataset$is_hound == 1 | dataset$is_bernard==1 | dataset$is_akita ==1 |
                                         dataset$is_borzoi==1 |  dataset$is_Beau==1 | dataset$is_Leon==1 )
        
        
        
        #### Gender #####
        male_names = as.character(read.table("male_names.txt")[,1])
        female_names = as.character(read.table("female_names.txt")[,1])
        Name = c(male_names,female_names)
        external_gender = c(rep(1,length(male_names)),rep(0,length(female_names)))
        gender_data = cbind.data.frame(Name,external_gender)
        gender_data=gender_data[!duplicated(gender_data$Name),]
        
        #### Name ####
        dataset$is_named = ifelse(dataset$Name=="",0,1)
        if(traintest == "trn"){
                dataset$y = as.factor(dataset$OutcomeType)
                dataset=dataset[order(dataset$AnimalID),]
        }else{
                dataset=dataset[order(dataset$ID),]
        }
        
        dataset = merge(dataset,gender_data,all.x = TRUE,by.x = "Name")
        dataset[!(dataset$external_gender %in% c(1,0)),"external_gender"]= 2
        if(traintest == "trn"){
                dataset=dataset[order(dataset$AnimalID),]
        }else{
                dataset=dataset[order(dataset$ID),]
        }
        
        dataset$not_human_name = rep(0,length(dataset$is_named))
        dataset$not_human_name[dataset$external_gender==2 & dataset$Name != ""] = 1
        dataset$name_ends_withy = endsWith(as.character(dataset$Name), "y")
        dataset$name_ends_withie = endsWith(as.character(dataset$Name), "ie")
        dataset$name_ends_with_y_or_ie = ifelse(dataset$name_ends_withy==1 | dataset$name_ends_withie ==1,1,0)
        
        ##########################################
        ## Add cheater data and see if it helps ##
        ##########################################
        dataset$sex_true = ifelse(grepl("Male",dataset$SexuponOutcome)==TRUE,2,ifelse(grepl("Female",dataset$SexuponOutcome)==TRUE,1,0))
        
        dataset$is_neutered = ifelse(grepl("Neutered",dataset$SexuponOutcome)==TRUE,1,0)
        dataset$is_spayed = ifelse(grepl("Spayed",dataset$SexuponOutcome)==TRUE,1,0)
        dataset$is_sterilized = ifelse(dataset$is_neutered ==1 | dataset$is_spayed == 1,1,0)
        
        dataset$length=as.numeric(unlist(lapply(strsplit(as.character(dataset$AgeuponOutcome)," "), function(x) x[1])))
        dataset$unit=unlist(lapply(strsplit(as.character(dataset$AgeuponOutcome)," "), function(x) x[2]))
        dataset$unit2=rep(0,length(dataset$unit))
        dataset$unit2[dataset$unit %in% c("week","weeks")] = 7
        dataset$unit2[dataset$unit %in% c("day","days")] = 1
        dataset$unit2[dataset$unit %in% c("month","months")] = 30
        dataset$unit2[dataset$unit %in% c("year","years")] = 365
        dataset$age = dataset$length*dataset$unit2/365
        dataset$age[is.na(dataset$age)] = 0
        dataset$age_unknown = (dataset$age == 0)
        dataset$length = NULL;dataset$unit = NULL
        
        dataset$date=unlist(lapply(strsplit(as.character(dataset$DateTime)," "), function(x) x[1]))
        dataset$time_of_day=unlist(lapply(strsplit(as.character(dataset$DateTime)," "), function(x) x[2]))
        dataset$hour_of_day=as.numeric(as.character(unlist(lapply(strsplit(dataset$time_of_day,":"), function(x) x[1]))))
        
        if(traintest == "trn"){
                dataset$month=as.numeric(unlist(lapply(strsplit(as.character(dataset$date),"/"), function(x) x[1])))
                dataset$year=as.numeric(unlist(lapply(strsplit(as.character(dataset$date),"/"), function(x) x[3])))
                dataset$day=weekdays(as.Date(dataset$date,format = "%m/%d/%Y"))
                weekend = (dataset$day %in% c("Friday","Saturday","Sunday"))
                
        }else if(traintest == "test"){
                dataset$month=as.numeric(unlist(lapply(strsplit(as.character(dataset$date),"-"), function(x) x[2])))
                dataset$year=as.numeric(unlist(lapply(strsplit(as.character(dataset$date),"-"), function(x) x[1])))  
                dataset$day=weekdays(as.Date(dataset$date,format = "%Y-%m-%d"))
                weekend = (dataset$day %in% c("Friday","Saturday","Sunday"))
                
        }
        dataset$date = NULL
        
        dmy <- dummyVars(~day, data = dataset,fullRank = FALSE)
        dataset_dummy <- data.frame(predict(dmy, newdata = dataset))
        dataset = cbind.data.frame(dataset,dataset_dummy)
        
        if(traintest =="trn"){
                traindata = dataset[,c("is_color_mix","nwords_color",
                                       "dayMonday","dayTuesday","dayWednesday","dayThursday","dayFriday","daySaturday","daySunday",
                                       "month","year","hour_of_day",
                                       "is_named","is_cat","is_sterilized","age","sex_true",
                                       "not_human_name","name_ends_with_y_or_ie",
                                       "is_tabby","is_mix","is_large_dog","is_chihuahua",
                                       "is_terrier","is_poodle","is_lab","is_schnau","is_retriever","is_pit",
                                       "is_miniature","is_collie","is_shepherd","is_beagle",
                                       "is_shorthair","is_mediumhair","is_siamese","is_longhair","is_tortie","is_torbie","is_point","is_tick",
                                       "y")]
                
        }else if(traintest == "test"){
                traindata = dataset[,c("is_color_mix","nwords_color",
                                       "dayMonday","dayTuesday","dayWednesday","dayThursday","dayFriday","daySaturday","daySunday",
                                       "month","year","hour_of_day",
                                       "is_named","is_cat","is_sterilized","age","sex_true",
                                       "not_human_name","name_ends_with_y_or_ie",
                                       "is_tabby","is_mix","is_large_dog","is_chihuahua",
                                       "is_terrier","is_poodle","is_lab","is_schnau","is_retriever","is_pit",
                                       "is_miniature","is_collie","is_shepherd","is_beagle",
                                       "is_shorthair","is_mediumhair","is_siamese","is_longhair","is_tortie","is_torbie","is_point","is_tick"
                )]
                
        }
        return(traindata)
}



trn_data = dataprep(trn,"trn")
test_data = dataprep(test,"test")
traindat = createDataPartition(trn_data$y, p = 0.8,list = FALSE)
trn_pp = trn_data[traindat,]
hld_pp = trn_data[-traindat,]
yindex = which(names(trn_pp)=="y")
new_y_trn = as.numeric(as.factor(trn_pp$y))-1
new_y_hld = as.numeric(as.factor(hld_pp$y))-1
#######################
#### Random Forest ####
#######################
tControl = trainControl(
        method = "cv",
        number = 5,
        classProbs = TRUE,
        summaryFunction = mnLogLoss,
        verboseIter = T)


tGrid <- expand.grid(  mtry = c(3,5,7,9,11),
                       splitrule = c("gini"),
                       min.node.size = c(4))

model_rf = train(y~.,
                       data = trn_pp,
                       trControl = tControl,
                       tuneGrid = tGrid,
                       method = "ranger",
                       metric = "logLoss",
                       importance = "impurity",
                       num.trees = 150)

model_rf_pred = predict(model_rf, hld_pp[,-which(names(hld_pp)=="y")])
model_rf_pred2=rep(0,length(model_rf_pred))
model_rf_pred2[model_rf_pred=="Adoption"] = 0
model_rf_pred2[model_rf_pred=="Died"] = 1
model_rf_pred2[model_rf_pred=="Euthanasia"] = 2
model_rf_pred2[model_rf_pred=="Return_to_owner"] = 3
model_rf_pred2[model_rf_pred=="Transfer"] = 4
levels(model_rf_pred2)= list("Adoption"="0","Died"="1","Euthanasia"="2","Return_to_owner"="3","Transfer"="4")

confuse_hld_rf = confusionMatrix(factor(model_rf_pred2),
                              factor(new_y_hld),
                              mode = "everything")


##################
#### Xgboost #####
##################
#### Cross Validation to find best Parameters ####

nround_cand = c(2000) 
eta_cand = c(0.005,0.01,0.015)
max_depth_cand=  c(5,7,9) #deeper trees (10,15) lead to overfit
colsample_bytree_cand = c(0.75)
subsample_cand = c(0.75)

best_minlogloss=100
nround_best=5
dtrn <- xgb.DMatrix(label = new_y_trn, data = as.matrix(trn_pp[,-yindex]))
dhld <- xgb.DMatrix(label = new_y_hld, data = as.matrix(hld_pp[,-yindex]))
watch <- list(train=dtrn, test = dhld)

for(i in 1:length(eta_cand)){
        for(j in 1:length(max_depth_cand)){
                xgb_cv_params <- list(objective = "multi:softprob",
                                   eval_metric = "mlogloss",
                                   num_class = 5,
                                   max_depth = max_depth_cand[j],
                                   subsample = subsample_cand,
                                   seed = 2,
                                   eta = eta_cand[i],
                                   colsample_bytree = colsample_bytree_cand ,
                                   min_child_weight =4)
                
                xgb_cv <- xgb.cv(data = dtrn, 
                               params = xgb_cv_params,
                               nround = nround_cand,
                               verbose =1,
                               print_every_n=100,
                               nfold = 5,
                               early_stopping_rounds = 50,
                               watch)
                
                plot(xgb_cv$evaluation_log[[2]],type = "l",
                     ylim = c(c(0.5,1.75)),
                     main = paste("max_depth_cand: ",max_depth_cand[j],
                                  "eta_cand: ", eta_cand[i]))
                lines(xgb_cv$evaluation_log[[4]],type = "l")
                text(x=200,y = 1.4,min(xgb_cv$evaluation_log[[4]]))
                
                minlogloss = min(xgb_cv$evaluation_log[[4]])
                minlogloss_iter = which(xgb_cv$evaluation_log[[4]] == minlogloss)
                
                if(minlogloss < best_minlogloss ){
                        best_minlogloss = minlogloss
                        xgb_best_params = xgb_cv_params
                        nround_best = minlogloss_iter
                }
        
        }

}
print(xgb_best_params)        
#### Run final model ####
xgb_final <- xgb.train(data = dtrn, 
                 params = xgb_best_params,
                 nround = nround_best+300,
                 verbose =1,
                 print_every_n=40,
                 watch)  
#Hld Predictions
y_pred <- predict(xgb_final, data.matrix(hld_pp[,-yindex]))
hld_preds = data.frame(matrix(y_pred, ncol = 5,byrow = TRUE))
hld_preds$rowmax <- apply(hld_preds, 1, FUN=max)
var = rep(7,nrow(hld_preds))
var[hld_preds$rowmax==hld_preds$X1] = 0
var[hld_preds$rowmax==hld_preds$X2] = 1
var[hld_preds$rowmax==hld_preds$X3] = 2
var[hld_preds$rowmax==hld_preds$X4] = 3
var[hld_preds$rowmax==hld_preds$X5] = 4

confuse_hld_xgb = confusionMatrix(factor(var),
                factor(new_y_hld),
                mode = "everything")

importance = xgb.importance(colnames(dtrn), model = xgb_final)

print(importance)


#### Test Prediction #####
test_pred <- predict(xgb_final, data.matrix(test_data))
test_pred2 = data.frame(matrix(test_pred, ncol = 5,byrow = TRUE))

test_pred2 = cbind.data.frame(test$ID,test_pred2)
names(test_pred2) = c("ID","Adoption","Died","Euthanasia","Return_to_owner","Transfer")
write.csv(test_pred2,"submission2.csv")


###################################
#### Neural net using mxnet   #####
###################################
TControl <- trainControl(method = 'cv',
                         number = 5,
                         classProbs = TRUE,
                         verboseIter = FALSE,
                         summaryFunction = mnLogLoss)
tGrid = expand.grid(layer1 = c(20),
                    layer2 = c(5),
                    layer3 = c(0),
                    beta1 = 0.9,
                    beta2 =0.999,
                    learningrate = c(0.01),
                    dropout = 0.1,
                    activation = "relu")
nn_fit2 <- train(y~.,
              data = trn_pp,
              method = 'mxnetAdam',
              preProcess = c('center', 'scale'),
              trControl = TControl,
              num.round = 25,
              tuneGrid=tGrid,
              na.action = na.exclude)
#hld_pp$age[is.na(hld_pp$age)] = 0

nn_fit_prob = predict(nn_fit2, hld_pp[,-which(names(hld_pp)=="y")],type = "prob")
nn_fit_class = predict(nn_fit2, hld_pp[,-which(names(hld_pp)=="y")])

nn_fit_class2=rep(0,length(nn_fit_class))
nn_fit_class2[nn_fit_class=="Adoption"] = "0"
nn_fit_class2[nn_fit_class=="Died"] = "1"
nn_fit_class2[nn_fit_class=="Euthanasia"] = "2"
nn_fit_class2[nn_fit_class=="Return_to_owner"] = "3"
nn_fit_class2[nn_fit_class=="Transfer"] ="4"
nn_fit_class2 = as.factor(nn_fit_class2)

confuse_hld_nn = confusionMatrix(factor(nn_fit_class2),
                                 factor(new_y_hld),
                                 mode = "everything")


########################
##### Toy Ensemble #####
########################
rf_pred = predict(model_rf, hld_pp[,-which(names(hld_pp)=="y")],type = "prob")
xgb_pred = data.frame(matrix(predict(xgb_final, data.matrix(hld_pp[,-yindex])), ncol = 5,byrow = TRUE))
nn_pred = predict(nn_fit2, hld_pp[,-which(names(hld_pp)=="y")],type = "prob")

weights = expand.grid(rf_wgt=seq(0,1,.1),xgb_wgt=seq(0,1,.1),nn_wgt=seq(0,1,.1))
weights = weights[rowSums(weights)<=1.02 & rowSums(weights)>=0.98,]
Accuracy= NULL
for(j in 1:nrow(weights)){
        ensemble_pred = rf_pred;for(i in 1:5){ensemble_pred[,i]=rep(0,nrow(ensemble_pred))}
        for(i in 1:5){
                ensemble_pred[,i] = (weights[j,1]*rf_pred[,i]+weights[j,2]*xgb_pred[,i]+weights[j,3]*nn_pred[,i])/3
        }
        ensemble_pred$rowmax <- apply(ensemble_pred, 1, FUN=max)
        var_ensemble = rep(7,nrow(ensemble_pred))
        var_ensemble[abs(ensemble_pred$rowmax-ensemble_pred$Adoption)<0.01] = 0
        var_ensemble[abs(ensemble_pred$rowmax-ensemble_pred$Died)<0.01] = 1
        var_ensemble[abs(ensemble_pred$rowmax-ensemble_pred$Euthanasia)<0.01] = 2
        var_ensemble[abs(ensemble_pred$rowmax-ensemble_pred$Return_to_owner)<0.01] = 3
        var_ensemble[abs(ensemble_pred$rowmax-ensemble_pred$Transfer)<0.01] = 4
        confuse_hld_ensemble = confusionMatrix(factor(var_ensemble),
                                      factor(new_y_hld),
                                      mode = "everything")
        Accuracy = c(Accuracy,confuse_hld_ensemble$overall[1])

        
        
        #print(paste("Accuracy:", confuse_hld_ensemble$overall[1], "/ W1:",weights[j,1],"  W2:",weights[j,2],"  W3:",weights[j,3]),sep = "")
}

final=cbind.data.frame(Accuracy,weights)




