rm(list = ls())    #delete objects
cat("\014")        #clear console

library(dplyr)
library(ggplot2)
library(glmnet)
library(tictoc)
library(MESS)
library(randomForest)
twdata          =       read.csv(file = 'C://Users//xy249//Desktop//TW.csv')
standard =function(x){
  x/sqrt(mean((x-mean(x))^2))
}

n               =       dim(twdata)[1]
p               =       dim(twdata)[2]-1
y               =       twdata$Flag
X               =       data.matrix(twdata[,-1])


#Function to calculate training and testing AUCs.
auc             =       function(train_prob_hat,test_prob_hat,y.train,y.test){
  dt                      =        0.01
  thta                    =        1-seq(0,1, by=dt)
  thta.length             =        length(thta)
  
  FPR.train               =        matrix(0, thta.length)
  TPR.train               =        matrix(0, thta.length)
  FPR.test                =        matrix(0, thta.length)
  TPR.test                =        matrix(0, thta.length)
  for (i in c(1:thta.length)){
    # calculate the FPR and TPR for train data 
    y.hat.train             =        ifelse(train_prob_hat > thta[i], 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train[i]            =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train[i]            =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    
    # calculate the FPR and TPR for test data 
    y.hat.test              =        ifelse(test_prob_hat> thta[i] , 1, 0)
    FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test                  =        sum(y.test==1) # total positives in the data
    N.test                  =        sum(y.test==0) # total negatives in the data
    FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    
    # print(paste("K=", K, " ki=",ki, ", K-fold CV=", Kfold.CV.err[i]))
  }
  auc.train     =       sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
  auc.test      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  
  #print(paste("train AUC =",sprintf("%.4f", lasso.auc.train)))
  #print(paste("test AUC  =",sprintf("%.4f", lasso.auc.test)))
  
  
  #errs.train      =   as.data.frame(cbind(FPR.train, TPR.train))
  #errs.train      =   data.frame(x=errs.train$V1,y=errs.train$V2,type="Train")
  #errs.test       =   as.data.frame(cbind(FPR.test, TPR.test))
  #errs.test       =   data.frame(x=errs.test$V1,y=errs.test$V2,type="Test")
  #errs            =   rbind(errs.train, errs.test)
  #c(errs,auc.train,auc.test)
  c(auc.train,auc.test)
}

##################################################
##################    2)     #####################
##################################################




# Split dataset and define initial dataframes
n.train         =       floor(0.9*n)
n.test          =       n-n.train

M                =     50
AUC.test.rf      =     rep(0,M)  # rf= randomForest
AUC.train.rf     =     rep(0,M)
AUC.test.el      =     rep(0,M)  #el = elastic net
AUC.train.el     =     rep(0,M)
AUC.test.rid     =     rep(0,M)
AUC.train.rid    =     rep(0,M)
AUC.test.lasso   =     rep(0,M) 
AUC.train.lasso  =     rep(0,M)
time.rid         =     rep(0,M)
time.lasso       =     rep(0,M)
time.rf          =     rep(0,M)
time.el          =     rep(0,M)


for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  weight           =     ifelse(y.train == 1, 1, sum(y.train)/(length(y.train)-sum(y.train)))
  start.time=Sys.time()
  #fit ridge and calculate and record the train and test AUCs
  cv.rid                =     cv.glmnet(X.train, y.train,weights = weight,alpha = 0, family = "binomial",intercept = T, type.measure = "auc", nfolds = 10)
  rid.fit               =     glmnet(X.train, y.train,weights = weight, alpha = 0, family = "binomial", intercept = T, lambda = cv.rid$lambda.min)
  rid.train.prob.hat    =     predict(rid.fit, newx = X.train,  type = "response")
  rid.test.prob.hat     =     predict(rid.fit, newx = X.test,  type = "response")
  rid.auc               =     auc(rid.train.prob.hat,rid.test.prob.hat,y.train,y.test)
  AUC.train.rid[m]      =     rid.auc[1]
  AUC.test.rid[m]       =     rid.auc[2]
  end.time=Sys.time()
  time.rid[m]=end.time-start.time
  
  
  start.time=Sys.time()
  #fit lasso and calculate and record the train and test AUCs
  cv.lasso                =     cv.glmnet(X.train, y.train,weights = weight, alpha = 1, family = "binomial",intercept = T, type.measure = "auc", nfolds = 10)
  lasso.fit               =     glmnet(X.train, y.train,weights = weight, alpha = 1, family = "binomial", intercept = T, lambda = cv.lasso$lambda.min)
  lasso.train.prob.hat    =     predict(lasso.fit, newx = X.train,  type = "response")
  lasso.test.prob.hat     =     predict(lasso.fit, newx = X.test,  type = "response")
  lasso.auc               =     auc(lasso.train.prob.hat,lasso.test.prob.hat,y.train,y.test)
  AUC.train.lasso[m]      =     lasso.auc[1]
  AUC.test.lasso[m]       =     lasso.auc[2]
  end.time=Sys.time()
  time.lasso[m]=end.time-start.time
  
  start.time=Sys.time()
  # fit elastic-net and calculate and record the train and test AUCs 
  a=0.5 # elastic-net
  cv.el                =     cv.glmnet(X.train, y.train,weights = weight, alpha = a, family = "binomial",intercept = T, type.measure = "auc", nfolds = 10)
  el.fit               =     glmnet(X.train, y.train, weights = weight,alpha = a, family = "binomial", intercept = T, lambda = cv.el$lambda.min)
  el.train.prob.hat    =     predict(el.fit, newx = X.train,  type = "response")
  el.test.prob.hat     =     predict(el.fit, newx = X.test,  type = "response")
  el.auc               =     auc(el.train.prob.hat,el.test.prob.hat,y.train,y.test)
  AUC.train.el[m]      =     el.auc[1]
  AUC.test.el[m]       =     el.auc[2]
  end.time=Sys.time()
  time.el[m]=end.time-start.time
  
  
  start.time=Sys.time()
  dat                  =     data.frame(x=X.train, y = as.factor(y.train))
  # fit RF and calculate and record the train and test AUCs 
  rf                   =     randomForest(y~.,data = dat, mtry = sqrt(p), importance = TRUE)
  rf.train.prob.hat    =     predict(rf, dat)
  dat.test             =     data.frame(x=X.test, y = as.factor(y.test))
  rf.test.prob.hat     =     predict(rf, dat.test)
  rf.train.num         =     as.numeric(rf.train.prob.hat)
  rf.test.num          =     as.numeric(rf.test.prob.hat)
  rf.auc               =     auc(rf.train.num,rf.test.num,y.train,y.test)
  AUC.train.rf[m]      =     rf.auc[1]
  AUC.test.rf[m]       =     rf.auc[2]
  end.time             =     Sys.time()
  time.rf[m]           =     end.time-start.time
  
  cat(sprintf("m=%3.f| AUC.test.rf=%.2f,  AUC.test.el=%.2f| AUC.train.rid=%.2f,  AUC.train.lasso=%.2f| \n", m,  AUC.test.rf[m], AUC.test.el[m],  AUC.train.rid[m], AUC.train.lasso[m]))
  
}
cat(sprintf('Ridge regression runing time:%4f \nLasso regression running time:%4f \nEl regression running time:%4f \nRandom forest running time:%4f \n', sum(time.rid),sum(time.lasso),sum(time.el),sum(time.rf)))



##########################################
##############  3)  ######################
##########################################
## b)
par(mfrow=c(1,2))
boxplot(AUC.train.rid,AUC.train.lasso,AUC.train.el,AUC.train.rf,names = c('Ridge','Lasso','Elastic','Rf'))
title('Traning AUCs')
boxplot(AUC.test.rid,AUC.test.lasso,AUC.test.el,AUC.test.rf,names = c('Ridge','Lasso','Elastic','Rf'))
title('Testing AUCs')



## c)
# We can use the last one which is 50th iteration

par(mfrow=c(3,1))

#par(mfrow=c(1,1))
#may show margins too large error, expand plot area to fix it.
plot(cv.lasso)
title("LASSO,10-fold CV time = 34.65s ",line = 3)
plot(cv.rid)
title("Ridge,10-fold CV time = 17.21s ",line = 3)
plot(cv.el)
title("Elastic,10-fold CV time = 40.91s ",line = 3)

weight           =     ifelse(y.train == 1, 1, sum(y.train)/(length(y.train)-sum(y.train)))
start.time=Sys.time()
cv_onesample.rid                =     cv.glmnet(X.train, y.train,weights = weight,alpha = 0, family = "binomial",intercept = T, type.measure = "auc", nfolds = 10)
end.time=Sys.time()
onesamplecvtime.rid = end.time-start.time

start.time=Sys.time()
cv_onesample.lasso              =     cv.glmnet(X.train, y.train,weights = weight,alpha = 1, family = "binomial",intercept = T, type.measure = "auc", nfolds = 10)
end.time=Sys.time()
onesamplecvtime.lasso = end.time-start.time

start.time=Sys.time()
cv_onesample.el                 =     cv.glmnet(X.train, y.train,weights = weight,alpha = 0.5, family = "binomial",intercept = T, type.measure = "auc", nfolds = 10)
end.time=Sys.time()
onesamplecvtime.el = end.time-start.time

cat(sprintf("One sample cv time:\n rid=%.2f,  lasso=%.2f| el=%.2f | \n", onesamplecvtime.rid, onesamplecvtime.lasso, onesamplecvtime.el))



time.lasso[50]
time.rid[50]
time.el[50]



## d)

sorted.auc.lasso    =    sort(AUC.test.lasso)
sorted.auc.rid      =    sort(AUC.test.rid)
sorted.auc.el       =    sort(AUC.test.el)
sorted.auc.rf       =    sort(AUC.test.rf)

# 90% test intervals
interval.lasso      =    sprintf('(%.2f,%.2f)',sorted.auc.lasso[3],sorted.auc.lasso[47])
interval.rid        =    sprintf('(%.2f,%.2f)',sorted.auc.rid[3],sorted.auc.rid[47])
interval.el         =    sprintf('(%.2f,%.2f)',sorted.auc.el[3],sorted.auc.el[47])
interval.rf         =    sprintf('(%.2f,%.2f)',sorted.auc.rf[3],sorted.auc.rf[47])


#Fit all the data

weight                =     ifelse(y == 1, 1, sum(y)/(length(y)-sum(y)))
start.time=Sys.time()
#fit ridge and calculate and record time
all.cv.rid            =     cv.glmnet(X, y,weights = weight,alpha = 0, family = "binomial",intercept = T, type.measure = "auc", nfolds = 10)

all.rid.fit           =     glmnet(X, y,weights = weight, alpha = 0, family = "binomial", intercept = T, lambda = all.cv.rid$lambda.min)
end.time=Sys.time()
all.time.rid=end.time-start.time


start.time=Sys.time()
#fit lasso and calculate and record time
all.cv.lasso            =     cv.glmnet(X, y,weights = weight,alpha = 1, family = "binomial",intercept = T, type.measure = "auc", nfolds = 10)
all.lasso.fit           =     glmnet(X, y,weights = weight, alpha = 1, family = "binomial", intercept = T, lambda = all.cv.lasso$lambda.min)
end.time=Sys.time()
all.time.lasso=end.time-start.time

start.time=Sys.time()
# fit elastic-net and calculate and record the time
a=0.5 # elastic-net
all.cv.el            =     cv.glmnet(X, y,weights = weight,alpha = a, family = "binomial",intercept = T, type.measure = "auc", nfolds = 10)
all.el.fit           =     glmnet(X, y,weights = weight, alpha = a, family = "binomial", intercept = T, lambda = all.cv.el$lambda.min)
end.time=Sys.time()
all.time.el=end.time-start.time


start.time=Sys.time()
dat                  =     data.frame(x=X, y = as.factor(y))
# fit RF and calculate and record the time
all.rf               =     randomForest(y~.,data = dat, mtry = sqrt(p), importance = TRUE)
end.time             =     Sys.time()
all.time.rf          =     end.time-start.time

cat(sprintf("all.time.lasso=%.2f,  all.time.rid=%.2f| all.time.el=%.2f,  all.time.rf=%.2f| \n", all.time.lasso, all.time.rid,  all.time.el, all.time.rf))



# Create 4x2 tables
ttables = matrix(c(interval.rid,all.time.rid,interval.lasso,all.time.lasso,interval.el,all.time.el,interval.rf,all.time.rf),ncol = 2,byrow = TRUE)
colnames(ttables) = c('90%interval','Time for all data')
rownames(ttables) = c('Ridge','Lasso','Elastic-Net','Random forest')
ttables






# Bar-plot

betaS.rf               =     data.frame(names(X[1,]), as.vector(all.rf$importance[,1]))
colnames(betaS.rf)     =     c( "feature", "value")

betaS.el               =     data.frame(names(X[1,]), as.vector(all.el.fit$beta))
colnames(betaS.el)     =     c( "feature", "value")

betaS.rid              =     data.frame(names(X[1,]), as.vector(all.rid.fit$beta))
colnames(betaS.rid)    =     c( "feature", "value")

betaS.lasso            =     data.frame(names(X[1,]), as.vector(all.lasso.fit$beta))
colnames(betaS.lasso)  =     c( "feature", "value")


# The order base on elastic net
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.el$value, decreasing = TRUE)])
betaS.el$feature     =  factor(betaS.el$feature, levels = betaS.el$feature[order(betaS.el$value, decreasing = TRUE)])
betaS.rid$feature    =  factor(betaS.rid$feature, levels = betaS.rid$feature[order(betaS.el$value, decreasing = TRUE)])
betaS.lasso$feature  =  factor(betaS.lasso$feature, levels = betaS.lasso$feature[order(betaS.el$value, decreasing = TRUE)])


elPlot =  ggplot(betaS.el, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  ggtitle('El importance of variables')

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  ggtitle('Rid importance of variables')

lassoPlot =  ggplot(betaS.lasso, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  ggtitle('Lasso importance of variables')

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  ggtitle('Rf importance of variables')


library(gridExtra)
grid.arrange(elPlot, ridPlot,lassoPlot,rfPlot, nrow = 4)


