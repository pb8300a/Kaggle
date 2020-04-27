#### Testing out whether we still know how to do shit or not, who knows?
library(caret)
library(data.table)
library(mltools)
library(plyr)
library(xgboost)

setwd("C:/Users/peter/Desktop/kaggle/houses")
trn = read.csv("train.csv",header = TRUE)
tst = read.csv("test.csv",header = TRUE) 


#################################
addNoAnswer = function(x){
  if(is.factor(x)) return(factor(x, levels=c(levels(x), "NA")))
  return(x)
}

onehot_custom = function(var,prefix){
  if(sum(is.na(var))>0){
    var = addNoAnswer(var)
    var[is.na(var)=="TRUE"] = "NA"
  }
  newvar = data.table(as.factor(var))
  encoded = one_hot(newvar)
  transformed = data.frame(encoded)
  names(transformed) = paste(prefix,substr(names(transformed),4,6),sep = "")
  return(transformed)
}

numeric_nas = function(dataset,variable,missvar){
  dataset[,missvar] = ifelse(is.na(dataset[,variable])==TRUE,1,0)
  dataset[is.na(dataset[,variable])==TRUE,variable] = mean(dataset[,variable],na.rm=TRUE)
  return(dataset)
}

convert_to_factors = function(dataset,list_of_names){
  for(i in 1:length(list_of_names)){
    dataset[,list_of_names[i]] = as.factor(dataset[,list_of_names[i]])
  }
  return(dataset)
}
################################
trn$log_sale_price = log(trn$SalePrice)
trn$SalePrice = NULL

data_prep = function(data2,datatype){
  #data2 = tst
  ### Divide dataset into numeric and character variables 
  data2 = convert_to_factors(data2,c("MoSold"))
  
  vartypes = sapply(data2,class)
  num_vars = data2[,vartypes %in% c("integer","numeric")];
  num_vars$Id = NULL
  
  fac_vars = data2[,!(vartypes %in% c("integer","numeric"))]
  num_vars=numeric_nas(num_vars,"LotFrontage","LotFrontage_miss")
  num_vars=numeric_nas(num_vars,"GarageYrBlt","GarageYrBlt_miss")
  num_vars=numeric_nas(num_vars,"GarageCars","GarageCars_miss")
  num_vars=numeric_nas(num_vars,"GarageArea","GarageArea_miss")
  num_vars=numeric_nas(num_vars,"MasVnrArea","MasVnrArea_miss")
  num_vars=numeric_nas(num_vars,"BsmtFullBath","BsmtFullBath_miss")
  num_vars=numeric_nas(num_vars,"BsmtHalfBath","BsmtHalfBath_miss")
  num_vars=numeric_nas(num_vars,"BsmtUnfSF","BsmtUnfSF_miss")
  num_vars=numeric_nas(num_vars,"TotalBsmtSF","TotalBsmtSF_miss")
  num_vars=numeric_nas(num_vars,"BsmtFinSF1","BsmtFinSF1_miss")
  num_vars=numeric_nas(num_vars,"BsmtFinSF2","BsmtFinSF2_miss")
  num_vars$Year2000 = apply(cbind.data.frame(
    rep(0,dim(num_vars)[1]),num_vars$YearBuilt-2000),1,max)
  ####one hot encode
  if(datatype == "tst"){fac_vars$MSZoning[is.na(fac_vars$MSZoning)=="TRUE"] = "RL"}
  
  zone = onehot_custom(fac_vars$MSZoning,"zone")[,-1]
  lshape = onehot_custom(fac_vars$LotShape,"lshape")[,-1]
  type = onehot_custom(fac_vars$BldgType,"type")[,-1]
  
  
  #we can do some recoding in order to lower the dimension of the final dataset to speed things up
  #recode housestyle
  fac_vars$HouseStyle=as.character(fac_vars$HouseStyle)
  fac_vars$HouseStyle[fac_vars$HouseStyle %in% c("SFoyer","2.5Unf","1.5Unf")] = "Other"
  fac_vars$HouseStyle[fac_vars$HouseStyle %in% c("2.5Fin")] = "2Story"
  style = onehot_custom(fac_vars$HouseStyle,"style")[,-1]
  
  fac_vars$Exterior1st[fac_vars$Exterior1st=="AsphShn"] = "AsbShng"
  fac_vars$Exterior1st[fac_vars$Exterior1st=="BrkComm"] = "CemntBd"
  fac_vars$Exterior1st[fac_vars$Exterior1st=="CBlock"] = "CemntBd"
  fac_vars$Exterior1st[fac_vars$Exterior1st=="ImStucc"] = "VinylSd"
  fac_vars$Exterior1st[fac_vars$Exterior1st=="Stone"] = "VinylSd"
  if(datatype == "tst"){fac_vars$Exterior1st[is.na(fac_vars$Exterior1st)=="TRUE"] = "VinylSd"}
  
  fac_vars$Exterior1st=droplevels(fac_vars$Exterior1st)
  ext1 = onehot_custom(fac_vars$Exterior1st,"ext1")[,-1]
  
  
  fac_vars$Exterior2nd[fac_vars$Exterior2nd=="AsphShn"] = "AsbShng"
  fac_vars$Exterior2nd[fac_vars$Exterior2nd=="Brk Cmn"] = "CmentBd"
  fac_vars$Exterior2nd[fac_vars$Exterior2nd=="CBlock"] = "CmentBd"
  fac_vars$Exterior2nd[fac_vars$Exterior2nd=="ImStucc"] = "VinylSd"
  fac_vars$Exterior2nd[fac_vars$Exterior2nd=="Other"] = "VinylSd"
  fac_vars$Exterior2nd[fac_vars$Exterior2nd=="Stone"] = "VinylSd"
  fac_vars$Exterior2nd = revalue(fac_vars$Exterior2nd,c("CmentBd"="CemntBd"))
  fac_vars$Exterior2nd=droplevels(fac_vars$Exterior2nd)
  if(datatype == "tst"){fac_vars$Exterior2nd[is.na(fac_vars$Exterior2nd)=="TRUE"] = "VinylSd"}
  ext2 = onehot_custom(fac_vars$Exterior2nd,"ext2")[,-1]
  vnr = onehot_custom(fac_vars$MasVnrType,"vnr")[,-1]
  extq = onehot_custom(fac_vars$ExterQual,"extq")[,-1]
  extc = onehot_custom(fac_vars$ExterCond,"extc")[,-1]
  
  fac_vars$Foundation[fac_vars$Foundation=="Stone"] = "CBlock"
  fac_vars$Foundation[fac_vars$Foundation=="Wood"] = "CBlock"
  fac_vars$Foundation=droplevels(fac_vars$Foundation)
  fac_vars$Foundation=droplevels(fac_vars$Foundation)
  
  found = onehot_custom(fac_vars$Foundation,"found")[,-1]
  bsmtq = onehot_custom(fac_vars$BsmtQual,"bsmtq")[,-1]
  
  fac_vars$BsmtCond[fac_vars$BsmtCond=="Po"] = "TA"
  fac_vars$BsmtCond=droplevels(fac_vars$BsmtCond)
  bsmtc = onehot_custom(fac_vars$BsmtCond,"bsmtc")[,-1]
  
  bsmte = onehot_custom(fac_vars$BsmtExposure,"bsmte")[,-1]
  bsmtf1 = onehot_custom(fac_vars$BsmtFinType1,"bsmtf1")[,-1]
  bsmtf2 = onehot_custom(fac_vars$BsmtFinType2,"bsmtf2")[,-1]
  
  fac_vars$HeatingQC[fac_vars$HeatingQC=="Po"] = "TA"
  fac_vars$HeatingQC=droplevels(fac_vars$HeatingQC)
  heatqc = onehot_custom(fac_vars$HeatingQC,"heatqc")[,-1]
  
  cent = onehot_custom(fac_vars$CentralAir,"cent")[,-1]
  
  fac_vars$Electrical[fac_vars$Electrical=="FuseP"] = "SBrkr"
  fac_vars$Electrical[fac_vars$Electrical=="Mix"] = "SBrkr"
  fac_vars$Electrical[is.na(fac_vars$Electrical)==TRUE] = "SBrkr"
  fac_vars$Electrical=droplevels(fac_vars$Electrical)
  elec = onehot_custom(fac_vars$Electrical,"elec")[,-1]
  
  if(datatype == "tst"){fac_vars$KitchenQual[is.na(fac_vars$KitchenQual)=="TRUE"] = "TA"}
  kitc = onehot_custom(fac_vars$KitchenQual,"kitc")[,-1]
  fire = onehot_custom(fac_vars$FireplaceQu,"fire")[,-1]
  gara = onehot_custom(fac_vars$GarageType,"gara")[,-1]
  garaf = onehot_custom(fac_vars$GarageFinish,"garaf")[,-1]
  #garaq = onehot_custom(fac_vars$GarageQual,"garaq")[,-1]
  #garac = onehot_custom(fac_vars$GarageCond,"garac")[,-1]
  pave = onehot_custom(fac_vars$PavedDrive,"pave")[,-1]
  fence = onehot_custom(fac_vars$Fence,"fence")[,-1]
  mosold = onehot_custom(fac_vars$MoSold,"mosold")[,-1]
  
  if(datatype == "tst"){fac_vars$SaleType[is.na(fac_vars$SaleType)=="TRUE"] = "WD"}
  salet = onehot_custom(fac_vars$SaleType,"salet")[,-1]
  salec = onehot_custom(fac_vars$SaleCondition,"salec")[,-1]
  
  
  one_hot_vars= cbind.data.frame(salet,salec,mosold,fence,pave,garaf,gara,fire,kitc,elec,
                                 cent,heatqc,bsmte,bsmtf1,bsmtf2,bsmtc,bsmtq,found,extc,
                                 extq,vnr,ext2,ext1,style,zone,lshape,type)
  
  final_data = cbind.data.frame(num_vars,one_hot_vars)
  return(final_data)
}

train_data = data_prep(trn,"trn")
tst_data = data_prep(tst,"tst")

train_data$log_sale_price = trn$log_sale_price
##### Correlation between numerics and y-variable
set.seed(12345)
trn_partition = createDataPartition(trn$log_sale_price,p = 0.8, list = FALSE)
trn_caret = train_data[trn_partition,]
hld_caret = train_data[-trn_partition,]
################################
### Benchmark Logit          ###
################################
tcontrol = trainControl(method = "cv", number = 5,verboseIter = TRUE)
ols_fit = train(log_sale_price~.,
                data = trn_caret, 
                method = "lm",
                trControl =tcontrol)
pred_hld = predict(ols_fit,hld_caret)
rmse_ols_hld = sqrt(mean((pred_hld - hld_caret$log_sale_price)^2))


################################
### LASSO                    ###
################################
number_cv = 10
tgrid = expand.grid(alpha = 1, lambda = seq(0,0.015,by = 0.0001))
#Set seeds to ensure RF is reproducable
set.seed(1231231)
seeds <- vector(mode = "list", length = number_cv+1)
for(i in 1:number_cv) seeds[[i]]<- sample.int(n=1000, nrow(tgrid))
seeds[[number_cv+1]] = sample.int(100,1)
tcontrol = trainControl(method = "cv",
                        number = number_cv,
                        verboseIter = TRUE)
lasso_fit = train(log_sale_price~.,
                data = trn_caret, 
                method = "glmnet",
                trControl =tcontrol,
                tuneGrid = tgrid)
#get lasso coefficients
lasso_coef=coef(lasso_fit$finalModel,lasso_fit$best$lambda)
pred_hld = predict(lasso_fit,hld_caret)
rmse_lasso_hld = mean(sqrt((pred_hld - hld_caret$log_sale_price)^2))
rmse_lasso_hld = sqrt(mean((pred_hld - hld_caret$log_sale_price)^2))

import = as.data.frame(varImp(lasso_fit)$importance)
rownames = row.names(import)
imp_vars = rownames[which(import$Overall>0)]
imp_vars=cbind.data.frame(imp_vars,import[import$Overall>0,])
names(imp_vars)[2] = "imp"
imp_vars[order(imp_vars$imp),]

pred_tst = exp(predict(lasso_fit,tst_data))
pred_tst = cbind.data.frame(tst$Id,pred_tst)
names(pred_tst) = c("Id","SalePrice")
write.table(pred_tst, file = "submission_v2.csv",row.names = FALSE)


#Score using a more constricting lasso
tgrid = expand.grid(alpha = 1, lambda = 0.009)
tcontrol = trainControl(method = "none")
lasso_fit_high_pnlty = train(log_sale_price~.,
                  data = trn_caret, 
                  method = "glmnet",
                  trControl =tcontrol,
                  tuneGrid = tgrid)
lasso_coef_high_pnlty=coef(lasso_fit_high_pnlty$finalModel,lasso_fit_high_pnlty$best$lambda)
pred_hld_high_pnlty = predict(lasso_fit_high_pnlty,hld_caret)
rmse_lasso_high_pnlty_hld = sqrt(mean((pred_hld_high_pnlty - hld_caret$log_sale_price)^2))



####### XGBOOST
nround_cand = 3000
eta_cand = c(0.02)
max_depth_cand = c(2,3)
colsample_bytree_cand = 0.75
subsample_cand = 0.7
min_child_weight_cand = c(2,3)

best_minrmse = 100
nround_best = 5
new_y_trn = trn_caret$log_sale_price
new_y_hld = hld_caret$log_sale_price
y_index = which(names(trn_caret)=="log_sale_price")

dtrn = xgb.DMatrix(label = new_y_trn, data = as.matrix(trn_caret[,-c(y_index)]))
dhld = xgb.DMatrix(label = new_y_hld, data = as.matrix(hld_caret[,-c(y_index)]))

watch = list(train=dtrn,test=dhld)
for(i in 1:length(max_depth_cand)){
  for(j in 1:length(min_child_weight_cand)){
    xgb_cv_params <- list(objective = "reg:linear",
                          eval_metric = "rmse",
                          eta = eta_cand,
                          max_depth = max_depth_cand[i],
                          colsample_bytree = colsample_bytree_cand ,
                          subsample = subsample_cand,
                          min_child_weight =min_child_weight_cand[j])
    set.seed(2134123)
    xgb_cv <- xgb.cv(data = dtrn, 
                     params = xgb_cv_params,
                     nround = nround_cand,
                     verbose =1,
                     print_every_n=50,
                     nfold = 5,
                     early_stopping_rounds = 50,
                     watch)
    
    plot(xgb_cv$evaluation_log[[4]] + xgb_cv$evaluation_log[[5]],type = "l",
         ylim = c(c(0,20)),
         main = paste("min_child_weight_cand: ",min_child_weight_cand[j],
                      "max_depth_cand: ", max_depth_cand[i]))
    text(x=200,y = 1.4,min(xgb_cv$evaluation_log[[4]]))
    
    #pick the parameters the give lowest (mean + SD) of RMSE
    rmse_plus_sd = xgb_cv$evaluation_log[[4]] + xgb_cv$evaluation_log[[5]]
    min_rmse = min(rmse_plus_sd)
    min_rmse_iter = which(rmse_plus_sd == min_rmse)
    
    if(min_rmse < best_minrmse ){
      best_minrmse = min_rmse
      xgb_best_params = xgb_cv_params
      nround_best = min_rmse_iter
    }
    
  }
  
}

print(xgb_best_params)        
#### Run final model ####
set.seed(123123)
xgb_final <- xgb.train(data = dtrn, 
                       params = xgb_best_params,
                       nround = nround_best,
                       verbose =1,
                       print_every_n=50,
                       watch)  
#Hld Predictions
hld_pred_xgb <- predict(xgb_final, dhld)
hld_rmse_xgb = sqrt(mean((hld_pred_xgb - hld_caret$log_sale_price)^2))
print(hld_rmse_xgb)
importance = xgb.importance(colnames(dtrn), model = xgb_final)
print(importance)


dtest = xgb.DMatrix(data = as.matrix(tst_data))
test_pred = exp(predict(xgb_final,newdata = dtest))
test_pred = cbind.data.frame(tst$Id,test_pred)
names(test_pred) = c("Id","SalePrice")
write.table(test_pred, file = "submission_v3.csv",row.names = FALSE)


###average of xgboost and lasso
avg_xgb_lass = exp((log(test_pred[,2])+log(pred_tst[,2]))/2)
avg_pred = cbind.data.frame(tst$Id,avg_xgb_lass)
names(avg_pred) = c("Id","SalePrice")
write.table(avg_pred, file = "submission_v4.csv",row.names = FALSE)
