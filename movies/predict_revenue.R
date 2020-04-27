#xgb3 adds budget_per_popularity, key_war, key_spo,genre_mus,prodcom_disn,prodcom_warn,key_art, country_cn, adds procom_none and runtime dummy
library(caret)
library(ranger)
library(stringr)
library(xgboost)

setwd("C:/Users/F400563/Desktop/uci/movies")

train = read.csv("train.csv", sep = ",")
test = read.csv("test.csv",sep = ",")

### try not trusting what looks like train with bad revenue
#train = train[train$revenue>100,]

y = train$revenue
train$revenue = NULL
alldat = rbind.data.frame(train,test)


names(alldat)[1]="id"

################################
##### Feature Extraction #######
################################

dum = rep(0,nrow(alldat))
dat = cbind.data.frame(alldat$id,dum)
dat$anthology = ifelse(alldat$belongs_to_collection == "",0,1)

dat$missing_budget = ifelse(alldat$budget==0,1,0)
dat$weird_budget = ifelse((alldat$budget<1000 & dat$missing_budget==0),1,0)

dat$log_budget =ifelse(dat$missing_budget==1,0,log(alldat$budget))
dat$popularity = alldat$popularity
dat$popularity[dat$popularity>50]=50   #cap for popularity
dat$runtime = alldat$runtime
dat$runtime[is.na(dat$runtime)] = 0
#dat$runtime_unknown = ifelse(dat$runtime==0,1,0)
#dat$released = ifelse(alldat$status == "Released",1,0)

dat$has_website = ifelse(alldat$homepage == "",0,1)
dat$has_tagline = ifelse(alldat$tagline=="",0,1)
#dat$has_overview = ifelse(alldat$overview=="",0,1)

#dat$tagline_size = nchar(as.character(alldat$tagline))
#dat$overview_size = nchar(as.character(alldat$overview))
#####genre#####
dat$genre_com = ifelse(grepl("Comedy",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_dra = ifelse(grepl("Drama",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_fam = ifelse(grepl("Family",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_rom = ifelse(grepl("Romance",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_cri = ifelse(grepl("Crime",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_thr = ifelse(grepl("Thriller",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_hor = ifelse(grepl("Horror",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_mys = ifelse(grepl("Mystery",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_sfi = ifelse(grepl("Science",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_wes = ifelse(grepl("Western",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_fan = ifelse(grepl("Fantasy",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_war = ifelse(grepl("War",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
#dat$genre_mus = ifelse(grepl("Music",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_adv = ifelse(grepl("Adventure",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_ani = ifelse(grepl("Animation",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_doc = ifelse(grepl("Documentary",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
#dat$genre_his = ifelse(grepl("History",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_tvm = ifelse(grepl("TV",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
dat$genre_act = ifelse(grepl("Action",alldat$genres,ignore.case = TRUE)==TRUE,1,0)
#dat$genre_no = ifelse(alldat$genres == "",1,0)
dat$genre_number = as.numeric(str_count(alldat$spoken_languages, "name"))

#####Language#####
dat$spoken_lang_num =  as.numeric(str_count(alldat$spoken_languages, "iso_639_1"))

dat$orig_lang_en = ifelse(alldat$original_language== "en",1,0)
dat$orig_lang_fr = ifelse(alldat$original_language== "fr",1,0)
dat$orig_lang_ja = ifelse(alldat$original_language== "ja",1,0)
dat$orig_lang_zh = ifelse(alldat$original_language== "zh",1,0)
#dat$orig_lang_de = ifelse(alldat$original_language== "de",1,0)
#dat$orig_lang_ko = ifelse(alldat$original_language== "ko",1,0)
#dat$orig_lang_ru = ifelse(alldat$original_language== "ru",1,0)
#dat$orig_lang_hi = ifelse(alldat$original_language== "hi",1,0)
#dat$orig_lang_es = ifelse(alldat$original_language== "es",1,0)
#dat$orig_lang_other = rep(0,nrow(dat))
#dat$orig_lang_other[dat$orig_lang_en == 0 & dat$orig_lang_fr == 0 & dat$orig_lang_ja == 0 & dat$orig_lang_zh == 0 ] = 1
dat$cast_size <- as.numeric(str_count(alldat$cast, "cast_id"))
####Some Keywords####
dat$key_cri = ifelse(grepl("kill",alldat$Keywords,ignore.case = TRUE)==TRUE,1,
                     ifelse(grepl("murder",alldat$Keywords,ignore.case = TRUE)==TRUE,1,
                            ifelse(grepl("crime",alldat$Keywords,ignore.case = TRUE)==TRUE,1,0)))
#dat$key_war = ifelse(grepl("war",alldat$Keywords,ignore.case = TRUE)==TRUE,1,0)
#dat$key_art = ifelse(grepl("art",alldat$Keywords,ignore.case = TRUE)==TRUE,1,0)
#dat$key_hol = ifelse(grepl("winter",alldat$Keywords,ignore.case = TRUE)==TRUE,1,
#                        ifelse(grepl("holiday",alldat$Keywords,ignore.case = TRUE)==TRUE,1,
#                                ifelse(grepl("christmas",alldat$Keywords,ignore.case = TRUE)==TRUE,1,0)))
#dat$key_chi = ifelse(grepl("kids",alldat$Keywords,ignore.case = TRUE)==TRUE,1,0)
#dat$key_lgbtq = ifelse(grepl("gay",alldat$Keywords,ignore.case = TRUE)==TRUE,1,
#              ifelse(grepl("lesbian",alldat$Keywords,ignore.case = TRUE)==TRUE,1,
#              ifelse(grepl("gender",alldat$Keywords,ignore.case = TRUE)==TRUE,1,
#              ifelse(grepl("queer",alldat$Keywords,ignore.case = TRUE)==TRUE,1,
#              ifelse(grepl("homosexual",alldat$Keywords,ignore.case = TRUE)==TRUE,1,
#              ifelse(grepl("bisexual",alldat$Keywords,ignore.case = TRUE)==TRUE,1,0))))))

dat$key_ind = ifelse(grepl("independent",alldat$Keywords,ignore.case = TRUE)==TRUE,1,0)

####release time#####
dat$release_month = as.numeric(unlist(lapply(strsplit(as.character(alldat$release_date),"/"), function(x) x[1])))
dat$release_day = as.numeric(unlist(lapply(strsplit(as.character(alldat$release_date),"/"), function(x) x[2])))
dat$release_year = as.numeric(unlist(lapply(strsplit(as.character(alldat$release_date),"/"), function(x) x[3])))
dat$release_year = ifelse(is.na(dat$release_year),0,dat$release_year)

#dat$release_year[dat$release_year>2018] = dat$release_year-100 #deals with cases of bad data
dat$release_day = ifelse(is.na(dat$release_day),0,dat$release_day)
dat$release_month = ifelse(is.na(dat$release_month),0,dat$release_month)

####Country of film#####
n =  strsplit(as.character(alldat$production_countries),",");countries = rep("",length(n))
for(i in 1:length(n)){
        countries[i] = paste(trimws(gsub("\\[|\\]|\\{|\\}|\\:|\\'|\\iso_3166_1", "",
                                         n[i][[1]][grepl("\\{",n[i][[1]])==TRUE])),collapse = " ")
}


#####country or origin####
dat$country_number = nchar(gsub(" ","",countries,fixed = TRUE))/2
dat$country_US = ifelse(grepl("US",countries,ignore.case = TRUE)==TRUE,1,0)
dat$country_CA = ifelse(grepl("CA",countries,ignore.case = TRUE)==TRUE,1,0)
#dat$country_GB = ifelse(grepl("GB",countries,ignore.case = TRUE)==TRUE,1,0)
dat$country_FR = ifelse(grepl("FR",countries,ignore.case = TRUE)==TRUE,1,0)
dat$country_JP = ifelse(grepl("JP",countries,ignore.case = TRUE)==TRUE,1,0)
#dat$country_CN = ifelse(grepl("CN",countries,ignore.case = TRUE)==TRUE,1,0)
dat$country_IN = ifelse(grepl("IN",countries,ignore.case = TRUE)==TRUE,1,0)
dat$country_DE = ifelse(grepl("DE",countries,ignore.case = TRUE)==TRUE,1,0)
dat$country_AU = ifelse(grepl("AU",countries,ignore.case = TRUE)==TRUE,1,0)
#dat$country_KR = ifelse(grepl("KR",countries,ignore.case = TRUE)==TRUE,1,0)
#dat$country_other = rep(0,nrow(dat))
#dat$country_other[dat$country_US == 0 & dat$country_CA == 0 & dat$country_GB == 0 & dat$country_FR == 0 &
#                  dat$country_JP == 0 & dat$country_IN == 0 & dat$country_DE == 0 & dat$country_AU == 0] = 1
dat$cast_size <- as.numeric(str_count(alldat$cast, "cast_id"))
dat$crew_size <- as.numeric(str_count(alldat$crew, "credit_id"))
#dat$bigdirector = ifelse(grepl("Spielberg",alldat$crew,ignore.case = TRUE)==TRUE,1,
#                  ifelse(grepl("James Cameron",alldat$crew,ignore.case = TRUE)==TRUE,1,0))

dat$prodcom_size = as.numeric(str_count(alldat$production_companies, "'id'"))
dat$prodcom_univ = ifelse(grepl("Universal",alldat$production_companies,ignore.case = TRUE)==TRUE,1,0)
#dat$prodcom_disn = ifelse(grepl("Disney",alldat$production_companies,ignore.case = TRUE)==TRUE,1,0)
dat$prodcom_para = ifelse(grepl("Paramount",alldat$production_companies,ignore.case = TRUE)==TRUE,1,0)
#dat$prodcom_warn = ifelse(grepl("Warner",alldat$production_companies,ignore.case = TRUE)==TRUE,1,0)
dat$prodcom_fox = ifelse(grepl("Fox",alldat$production_companies,ignore.case = TRUE)==TRUE,1,0)
#dat$prodcom_mira = ifelse(grepl("Miramax",alldat$production_companies,ignore.case = TRUE)==TRUE,1,0)
#dat$prodcom_marv = ifelse(grepl("Marvel",alldat$production_companies,ignore.case = TRUE)==TRUE,1,0)
dat$prodcom_none = ifelse(alldat$production_companies=="",1,0)

####budget efficiency
dat$budget_per_crewmem = ifelse(dat$crew_size == 0,0,dat$log_budget/dat$crew_size)
dat$budget_per_castmem = ifelse(dat$cast_size == 0,0,dat$log_budget/dat$cast_size)
dat$budget_per_popularity= ifelse(dat$cast_size == 0,0,dat$log_budget/dat$popularity)
dat$budget_per_runtime= ifelse(dat$runtime == 0,0,dat$log_budget/dat$runtime)

#####cast gender
newvar = gsub("\\'", "",alldat$cast)

dat$gender0 = as.numeric(str_count(newvar,"gender: 0"))
dat$gender1 = as.numeric(str_count(newvar,"gender: 1"))
dat$gender2 = as.numeric(str_count(newvar,"gender: 2"))
dat$gender0_rate = dat$gender0/(dat$gender0+dat$gender1+dat$gender2)
dat$gender1_rate = dat$gender1/(dat$gender0+dat$gender1+dat$gender2)
dat$gender2_rate = dat$gender2/(dat$gender0+dat$gender1+dat$gender2)

### Sequel or Trilogy movie###
#dat$sequel = ifelse(grepl(" 2",alldat$original_title,ignore.case = TRUE)==TRUE,1,0)
#dat$tri = ifelse(grepl(" 3",alldat$original_title,ignore.case = TRUE)==TRUE,1,0)
#dat$threed = ifelse(grepl("3D",alldat$original_title,ignore.case = TRUE)==TRUE,1,
#                    ifelse(grepl("3-D",alldat$original_title,ignore.case = TRUE)==TRUE,1,0))
#dat$tri[dat$threed==1] = 0
#dat$threed = ifelse(grepl("3d",alldat$Keywords,ignore.case = TRUE)==TRUE,1,0)

names(dat)[1] = "id"
dat$dum = NULL


train_clean = dat[1:nrow(train),]
test_clean = dat[-c(1:nrow(train)),]
train_clean$log_revenue = log(y)
set.seed(12345)
traindat = createDataPartition(train_clean$log_revenue, p = 0.8,list = FALSE)
trn = train_clean[traindat,]
hld = train_clean[-traindat,]


##########################################################################################################
##########################################################################################################
# 
# #############################
# ####### Basic Model #########
# number_cv = 5
# tuneGrid=expand.grid(alpha=1,lambda=seq(0, 0.04, by = 0.001))
# #Set seeds to ensure RF is reproducable
# set.seed(1231231)
# seeds <- vector(mode = "list", length = number_cv+1)
# for(i in 1:number_cv) seeds[[i]]<- sample.int(n=1000, nrow(tuneGrid))
# seeds[[number_cv+1]] = sample.int(100,1)
# 
# tControl2 = trainControl(method = "cv",
#                          number = number_cv,
#                          summaryFunction=defaultSummary,
#                          verboseIter  = TRUE,
#                          seeds = seeds)
# 
# lasso_all <- train(log_revenue ~ .,
#                    data=trn[,-c(1,2)],
#                    method="glmnet",
#                    family="gaussian",
#                    trControl  = tControl2,
#                    tuneGrid = tuneGrid)
# 
# hld_pred = predict(lasso_all, newdata = hld)
# hld_rmse = mean(sqrt((hld_pred - hld$log_revenue)^2))
# 
# 
# ###################################
# ####### Random Forest Model #######
# ###################################
# number_cv = 5
# num_tree=100
# tGrid <- expand.grid( mtry = seq(5,20,by = 5),splitrule = 'variance',min.node.size = seq(10,40, by = 10))
# #Set seeds to ensure RF is reproducable
# set.seed(1231231)
# seeds <- vector(mode = "list", length = number_cv+1)
# for(i in 1:number_cv) seeds[[i]]<- sample.int(n=1000, nrow(tGrid))
# seeds[[number_cv+1]] = sample.int(100,1)
# 
# tControl2 = trainControl(method = "cv",
#                          number = number_cv,
#                          summaryFunction=defaultSummary,
#                          verboseIter  = TRUE,
#                          seeds = seeds)
# 
# rf_all <- train(log_revenue ~ .,
#                    data=trn[,-c(1,2)],
#                    method = "ranger",
#                    trControl  = tControl2,
#                    tuneGrid = tGrid,
#                    importance = 'impurity',
#                    num.trees = num_tree
# )
# 
# hld_pred = predict(rf_all, newdata = hld)
# hld_rmse = mean(sqrt((hld_pred - hld$log_revenue)^2))
# ####Predict for test set#####
# test_pred_rf = predict(rf_all, newdata = test_clean)
# submit = cbind.data.frame(test$ï..id,exp(test_pred_rf))
# names(submit)=c("id","revenue")
# write.csv(submit,"submission_rf2.csv",row.names = FALSE)
#############################
####### XGBoost Model #######
#############################


nround_cand = c(2000) 
eta_cand = c(0.0065) #seq(0.01,0.015,by = 0.001)
max_depth_cand=  c(5,6,7) #deeper trees (10,15) lead to overfit
colsample_bytree_cand = 0.7#seq(0.5,1,by = 0.1)
subsample_cand = 0.7#seq(0.5,1,by = 0.1)
min_child_weight_cand = c(3)

best_minrmse=100
nround_best=5
new_y_trn = trn$log_revenue
new_y_hld= hld$log_revenue
yindex = which(names(trn)=="log_revenue")
idindex = which(names(trn)=="id")

dtrn <- xgb.DMatrix(label = new_y_trn, data = as.matrix(trn[,-c(yindex,idindex)]))
dhld <- xgb.DMatrix(label = new_y_hld, data = as.matrix(hld[,-c(yindex,idindex)]))
watch <- list(train=dtrn, test = dhld)

for(i in 1:length(min_child_weight_cand)){
        for(j in 1:length(max_depth_cand)){
                xgb_cv_params <- list(objective = "reg:linear",
                                      eval_metric = "rmse",
                                      eta = eta_cand,
                                      max_depth = max_depth_cand[j],
                                      colsample_bytree = colsample_bytree_cand ,
                                      subsample = subsample_cand,
                                      min_child_weight =min_child_weight_cand[i])
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
                     main = paste("max_depth_cand: ",max_depth_cand[j],
                                  "eta_cand: ", eta_cand[i]))
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
hld_rmse_xgb = mean(sqrt((hld_pred_xgb - hld$log_revenue)^2))
print(hld_rmse_xgb)
importance = xgb.importance(colnames(dtrn), model = xgb_final)
print(importance)


resid =hld_pred_xgb - hld$log_revenue

####Predict for test set#####
dtest <- xgb.DMatrix(data = as.matrix(test_clean[,-1]))
test_pred = predict(xgb_final, newdata = dtest)
submit = cbind.data.frame(test$ï..id,exp(test_pred))
names(submit)=c("id","revenue")
write.csv(submit,"submission_xgb3.csv",row.names = FALSE)


