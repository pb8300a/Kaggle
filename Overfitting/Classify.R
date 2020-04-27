library(caret)
library(pROC)
library(glmnet)
library(tidyverse)
setwd("C:/Users/F400563/Desktop/uci/overfit")

#Load the data
trn = read.csv("train.csv")
test = read.csv("test.csv")

#Center and Scale Variables
#trn[,c(3:(ncol(trn)-1))]=scale(trn[,c(3:(ncol(trn)-1))], scale = TRUE, center = TRUE)
#test[,c(2:(ncol(test)-1))]=scale(test[,c(2:(ncol(test)-1))], scale = TRUE, center = TRUE)

#Output variables -- 36% 0, 64% 1
table(trn$target)

#Input variables -- they are basically all mean = 0, std = 1 
var_means = colMeans(trn[,-c(1,2)])
std_means = NULL
for(i in 3:ncol(trn)){std_means = c(std_means,sqrt(var(trn[,i])))}


# What is the relationship between different variables and the Target
corr = NULL;
for(i in 3:nrow(trn)){
       corr = c(corr,cor(trn$target,trn[,i]))
}
par(mfrow = c(1,1))
plot(corr)

pos_correlation = c(length(corr[corr>0]),length(corr[corr<0]))
high_correlation = names(trn)[which(abs(corr)>0.15)+2] #add two to account for the shift above ID and Target

par(mfrow = c(1,3))
for(i in high_correlation){
      boxplot(trn[,i]~as.factor(trn$target))
}

par(mfrow = c(1,1))


trn$y_char = ifelse(trn$target==1,"Y_One","Y_Zero")

######################################
### Lasso - Check Many Regularizations
######################################
number_cv = 5
tuneGrid=expand.grid(alpha=1,lambda=seq(0, 0.25, by = 0.001))
#Set seeds to ensure RF is reproducable
set.seed(1231231)
seeds <- vector(mode = "list", length = number_cv+1)
for(i in 1:number_cv) seeds[[i]]<- sample.int(n=1000, nrow(tuneGrid))
seeds[[number_cv+1]] = sample.int(100,1)

tControl2 = trainControl(method = "cv",
                         number = number_cv,
                         classProbs = TRUE ,
                         summaryFunction=twoClassSummary,
                         verboseIter  = TRUE,
                         seeds = seeds)

lasso_all <- train(y_char ~ .,
                data=trn[,-c(1,2)],
                method="glmnet",
                family="binomial",
                metric = "ROC",
                trControl  = tControl2,
                tuneGrid = tuneGrid)

#### See how much we can regularize without losing too much performance 
plot(lasso_all)
#Two spots at which we can regularize while keeping performance are (0.144, 0.06)


######################################
# Check at lambda = 0.05
######################################

tuneGrid=expand.grid(alpha=1,lambda=0.05)
tControl2 = trainControl(method = "none",classProbs = TRUE ,summaryFunction=twoClassSummary)
lasso_05 <- train(y_char ~ .,
                data=trn[,-c(1,2)],
                method="glmnet",
                family="binomial",
                metric = "ROC",
                trControl  = tControl2,
                tuneGrid = tuneGrid)
#note the overlap between high_correlation variables and lasso selected variables
high_correlation
import = as.data.frame(varImp(lasso_05)$importance)
rownames = row.names(import)
imp_vars = rownames[which(import$Overall>0)]


pred_trn = predict(lasso_05, newdata = trn,type ="prob")
pROC::auc(roc(trn$target,pred_trn[,2]))

test_prediction1 = predict(lasso_05, newdata = test,type ="prob")
test_prediction1 = cbind.data.frame(test$id,test_prediction1$Y_One)
names(test_prediction1) = c("id","target")
#write.csv(test_prediction1,"submission_lasso2.csv",row.names = FALSE) 

#####################################################################################
#### Try a Random Forest, but use only variables LASSO identified as important ####
#####################################################################################


set.seed(3456)
traindat = createDataPartition(trn$y, p = 1,list = FALSE)
trn_pp = trn[traindat,]
hld_pp = trn[-traindat,]
par(mar = c(2,3,4,1))
for(numcomp in seq(115,115,by = 10)){
        number_cv = 10
        number_repeat = 5
        set.seed(1231231)
        imp_var_index = which(names(trn_pp) %in% imp_vars) 
        preProc <- preProcess(trn_pp[,-c(1,2,imp_var_index,303)],method="pca",pcaComp = numcomp) #pcaComp = x --> this messes up phat
        trn_pca <- predict(preProc,trn_pp[,-c(1,2,imp_var_index,303)])
        trn_pca = cbind.data.frame(trn_pp$y_char,trn_pca,trn_pp[,imp_vars])
        names(trn_pca)[1] = "target"
        
        
        tuneGrid=expand.grid(alpha=1,lambda=seq(0,0.2,by =0.001))
        
        set.seed(1231231)
        seeds <- vector(mode = "list", length = (number_cv*number_repeat))
        for(i in 1:c(number_cv*number_repeat)) seeds[[i]]<- sample.int(n=10000, nrow(tuneGrid))
        seeds[[(number_cv*number_repeat)+1]] = sample.int(1000,1)
        
        
        tControl = trainControl(method = "repeatedcv",
                                number = number_cv, 
                                repeats = number_repeat,
                                classProbs = TRUE,
                                seeds = seeds,
                                summaryFunction = twoClassSummary,
                                verboseIter = F)
        
        lasso_pca <- train(target ~ .,
                           data=trn_pca,
                           method="glmnet",
                           family="binomial",
                           metric = "ROC",
                           trControl  = tControl,
                           tuneGrid = tuneGrid)
        #note the overlap between high_correlation variables and lasso selected variables
        import_pca = as.data.frame(varImp(lasso_pca)$importance)
        rownames_pca = row.names(import_pca)
        imp_vars_pca = rownames_pca[which(import_pca$Overall>0)]
        
        
        trn_predictions_pca = predict(lasso_pca, newdata = trn_pca,type ="prob")[,2]
        print(pROC::auc(roc(trn_pca$target,trn_predictions_pca)))
        
        #hld_pca <- predict(preProc,hld_pp[,-c(1,2,imp_var_index,303)])
        #hld_pca = cbind.data.frame(hld_pp$target,hld_pca,hld_pp[,imp_vars])
        #names(hld_pca)[1]="target"
        #hld_predictions_pca = predict(lasso_pca, newdata = hld_pca,type ="prob")[,2]
        #print(pROC::auc(roc(hld_pca$target,hld_predictions_pca)))
        
        print(numcomp)
        d=plot(lasso_pca)
        plot(d)
        test_pca <- predict(preProc,test)
        test_pca = cbind.data.frame(test_pca,test)
        
        test_predictions_pca = 1-predict(lasso_pca, newdata = test_pca,type ="prob")[,2]
        
        
        
        test_prediction_pca_data = cbind.data.frame(test$id,test_predictions_pca)
        names(test_prediction_pca_data) = c("id","target")
        hist(test_prediction_pca_data$target)
        print(test_prediction_pca_data[1:10,])
        
        
}

