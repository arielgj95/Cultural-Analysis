library(dplyr)
library(randomForest)
library(caret)
library(ggplot2)
library(grid) 
library(gridExtra)
library(pROC)
library(MLmetrics)
library(e1071)
library(RJSONIO)
library(doParallel)
library(foreach)
library(binaryLogic)
library(mltools)
library(data.table)
#registerDoMC()


no_cores <- detectCores()  # Number of cores
cl<-makeCluster(no_cores) #4
registerDoParallel(cl)

###### Workspace initialization
rm(list=ls())
cat("\014")
set.seed(11)
graphics.off()    #used to avoid problems during plots
options(max.print=1000000)  #print all the data


min_max_norm <- function(x, na.rm = FALSE){   #na values are already removed
  
  di <- max(x) - min(x)
  if(di > 1e-6)                               #normalize only if difference between max and min is > 1e-6 to avoid numerical issues
    return((x- min(x)) /(max(x)-min(x)))
  else
    return(0)
}

encode_binary <- function(x, order = unique(x), name = NULL){

  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x2 <- as.binary(x)
  maxlen <- max(sapply(x2, length))
  x2 <- lapply(x2, function(y) {
    l <- length(y)
    if (l < maxlen) {
      y <- c(rep(0, (maxlen - l)), y)
    }
    y
  })
  d <- as.data.frame(t(as.data.frame(x2)))
  rownames(d) <- NULL
  colnames(d) <- paste0(paste(name,"_",sep=""), 1:maxlen)
  return(d)
}

encode_OneHot <- function(x,name=NULL){
  
  x <- as.factor(x)
  d <- one_hot(as.data.table(x))
  return(d)
}


normalization <- function(data, norm_type, conc=FALSE){
  
  for(i in 1:ncol(data)){                   #normalize all the columns (in this case there are 3 columns to be normalized)
    colname <- colnames(data)[[i]]
    if(!conc){
      data[,i] <- norm_type(data[,i])
    }
    else{
      #print(unique(data[,i]))
      d_norm <- norm_type(data[,i],name=colname)
      if(i==1){
        df <- data.frame(row.names=1:nrow(data))  ##empty dataframe
        d <- cbind(df,d_norm)
      }
      else
        d<-cbind(d,d_norm)
    }
  }
  if(!conc)
    return(data)
  else
    return(d)
}


read_data_food <- function(data, country1, country2){              ### This function is the only one that depends from dataset
  
  len_x_col <- 50
  len_y_col <- 8
  data <- na.omit(data)                                                                         #remove all data with empty cells
  #print(colnames(data))
  data <- filter(data, cuisine == country1 | cuisine == country2)
  data[,len_x_col+1:len_y_col] <- ifelse(data[,len_x_col+1:len_y_col] == 1,"Present","Not Present")
  
  for(y in (len_x_col+1):(len_x_col+len_y_col))    #from string to factor variables
    data[,y] <- factor(data[,y])          
  
  data$cuisine <- factor(data$cuisine)                                          #from string to factor variables

  data_country1 <- select(filter(data, cuisine == country1),-cuisine)                            #take the data of one country
  data_country2 <- select(filter(data, cuisine == country2),-cuisine)                            #take the data of the other country
  #print(head(data_country1,10))
  
  return(list("data"=data,"data_country1"=data_country1,"data_country2"=data_country2))
}


read_data_loans <- function(data, country1, country2, normalize=TRUE, norm_fun=normalization){              ### This function is the only one that depends from dataset
  
  x_col <- c(2:5,10,11,15,16,18,19,8)
  data <- data[x_col]
  data <- na.omit(data)                                                                         #remove all data with empty cells
  male_borr <- female_borr <- rep(0,length(data))

  for(i in 1:nrow(data)){
    gender_table <- table(strsplit(data[i,"borrower_genders"], split = ", "))
    gender_names <- names(gender_table)
    if("male" %in% gender_names)
      male_borr[[i]] <- table(strsplit(data[i,"borrower_genders"], split = ", "))[["male"]]  # How many male borrowers are present for each loan
    else
      male_borr[[i]] <- 0
    if("female" %in% gender_names)
      female_borr[[i]] <- table(strsplit(data[i,"borrower_genders"], split = ", "))[["female"]] # How many female borrowers are present for each loan
    else
      female_borr[[i]] <- 0
  }
  data <- cbind(data, male_borrowers = male_borr)
  data <- cbind(data, female_borrowers = female_borr)
  data <- subset(data, select = -borrower_genders)
  data <- filter(data, country == country1 | country == country2)
  data$country <- factor(data$country)                                          #from string to factor variables
  
  is_funded <- ifelse(data['funded_amount'] >= data['loan_amount'],1,0) # add a column called is_founded which is 1 if 'funded_amount' = 'loan_amount', 0 otherwise
  colnames(is_funded) <- "is_funded"
  data <- cbind(data, is_funded = is_funded)  
  data[,"is_funded"] <- ifelse(data[,"is_funded"] == 1,"Funded","Not Funded")  #transform to factor variable
  data[,"is_funded"] <- factor(data[,"is_funded"]) #from string to factor variables
  
  if(normalize){   
    data[,c("funded_amount","loan_amount","term_in_months","lender_count","male_borrowers","female_borrowers")] <- #normalize all the data so that it can be used by the models
      norm_fun(data[,c("funded_amount","loan_amount","term_in_months","lender_count","male_borrowers","female_borrowers")],min_max_norm, conc=FALSE)                
    bin_feat <- norm_fun(data[,c("activity","partner_id","currency")],encode_binary, conc=TRUE)
    onehot_feat <- norm_fun(data[,c("repayment_interval","sector")],encode_OneHot, conc=TRUE)
    data <- cbind(data, c(bin_feat,onehot_feat))
    data <- subset(data, select = -c(activity,partner_id,repayment_interval,sector,currency))
  }
  data_country1 <- select(filter(data, country == country1),-country)                            #take the data of one country
  data_country2 <- select(filter(data, country == country2),-country)                            #take the data of the other country
  #print(head(data_country1,10))
  
  return(list("data"=data,"data_country1"=data_country1,"data_country2"=data_country2))
}


extract_sets <- function(data, n_tr, n_val){

  samp <- sample(nrow(data), n_tr + n_val)                    #sample random rows of the dataset for training and validation
  #build training,validation and test sets 
  XTr <- data[samp[1:n_tr], c(-1)]
  XVal <- data[samp[(n_tr+1):(n_tr + n_val)], c(-1)]
  XTe <- data[-samp, c(-1)]
  YTr <- data[samp[1:n_tr], c(1)]
  YVal <- data[samp[(n_tr+1):(n_tr + n_val)], c(1)]
  YTe <- data[-samp, c(1)]
  
  return(list("XTr"=XTr,"XVal"=XVal,"XTe"=XTe,
              "YTr"=YTr, "YVal"=YVal, "YTe"=YTe))
  
}


choose_TrValTe_sets <- function(model, data1, data2, all_models){   #depending on the model used, we will have different training and validation sets
  
  if(model == all_models[[1]]){
    XTr <- data1[["XTr"]]
    XVal <- data1[["XVal"]]
    YTr <- data1[["YTr"]]
    YVal <- data1[["YVal"]]
  }
  else if(model == all_models[[2]]){
    XTr <- data2[["XTr"]]
    XVal <- data2[["XVal"]]
    YTr <- data2[["YTr"]]
    YVal <- data2[["YVal"]]
  }
  else if(model == all_models[[3]]){
    XTr <- rbind(data1[["XTr"]], data2[["XTr"]])
    XVal <- rbind(data1[["XVal"]], data2[["XVal"]])
    YTr <- c(as.array(data1[["YTr"]]), as.array(data2[["YTr"]]))
    YVal <- c(as.array(data1[["YVal"]]), as.array(data2[["YVal"]]))
  }
  
  else if(model == all_models[[4]]){    
    #Add a column of 0 if data depends from one country, 1 if it depends from the other
    data1[["XTr"]] <- cbind(data1[["XTr"]], country_lab = rep(0,nrow(data1[["XTr"]])))
    data1[["XVal"]] <- cbind(data1[["XVal"]], country_lab = rep(0,nrow(data1[["XVal"]])))
    data1[["XTe"]] <- cbind(data1[["XTe"]], country_lab = rep(0,nrow(data1[["XTe"]])))
    data2[["XTr"]] <- cbind(data2[["XTr"]], country_lab = rep(1,nrow(data2[["XTr"]])))
    data2[["XVal"]] <- cbind(data2[["XVal"]], country_lab = rep(1,nrow(data2[["XVal"]])))
    data2[["XTe"]] <- cbind(data2[["XTe"]], country_lab = rep(1,nrow(data2[["XTe"]])))
    
    XTr <- rbind(data1[["XTr"]], data2[["XTr"]])
    XVal <- rbind(data1[["XVal"]], data2[["XVal"]])
    YTr <- c(as.array(data1[["YTr"]]), as.array(data2[["YTr"]]))
    YVal <- c(as.array(data1[["YVal"]]), as.array(data2[["YVal"]]))
  }
  
  return(list("XTr"=XTr,"XVal"=XVal,"YTr"=YTr,"YVal"=YVal,"XTe1"=data1[["XTe"]],"XTe2"=data2[["XTe"]],
              "XVal1"=data1[["XVal"]],"XVal2"=data2[["XVal"]]))
}


rf_model <- function(xtr, ytr, xte1, xte2, n_trees, mtry){
  
  
  cl<-makePSOCKcluster(6)
  
  registerDoParallel(cl)
  
  #M <- foreach(mtry = mtry_vec, .combine=randomForest::combine,
  #              .multicombine=TRUE, .packages='randomForest') %dopar% {
  #                randomForest(x = xtr, y = ytr, importance = TRUE, ntree = n_trees,       #build the random forest model and train it using the tr_data and tuple of params
  #                             sampsize = c(min(summary(ytr)),min(summary(ytr))),
  #                             mtry = mtry_vec)
  #              }
  M <- prandomForest(x = xtr, y = ytr, importance = TRUE, ntree = n_trees,       #build the random forest model and train it using the tr_data and tuple of params
                    sampsize = c(min(summary(ytr)),min(summary(ytr))),
                    mtry = mtry)
  
  #compute predictions and probabilities
  Y1 <- predict(M,xte1)
  Y2 <- predict(M,xte2)
  Y1_prob <- as.data.frame(predict(M,xte1, type = "prob")) 
  Y2_prob <- as.data.frame(predict(M,xte2, type = "prob")) 
  
  stopCluster(cl)
  return(list("res1"=Y1,"res2"=Y2,"res1_prob"=Y1_prob, "res2_prob"=Y2_prob))
}


svm_model <- function(xtr,ytr,xte1,xte2,gamma,cost,class1,class2){
  
  unb_fact <- table(c(ytr))[[class1]]/table(c(ytr))[[class2]]                #compute unbalance between classes 
  M <- svm(x = xtr, y = ytr, kernel = "radial",                                 #build the svm model and train it using the tr_data and tuple of params
           gamma = gamma, cost = cost,
           probability=TRUE, class.weights = c(class1=1/(1+unb_fact),class2=1-(1/(1+unb_fact))))    ######Savory then Sweet
  #compute predictions and probabilities
  Y1 <- predict(M,xte1)
  Y2 <- predict(M,xte2)
  Y1_prob <- as.data.frame(attr(predict(M,xte1,probability=TRUE),"probabilities")) 
  Y2_prob <- as.data.frame(attr(predict(M,xte2,probability=TRUE),"probabilities")) 
  
  return(list("res1"=Y1,"res2"=Y2,"res1_prob"=Y1_prob, "res2_prob"=Y2_prob))
}


init_final_data <- function(Val_data,Te_data,YVa1,YVa2,YTe1,YTe2,n_loops,classes,p,res){  #Val_data contains Y predicted by using validation sets
                                                                                          #Te_data contains Y predicted by using test sets
  ##### Vectors initialization                                                            #YV1e,YV2,YT1,YT2 contain real data used for validation and test
  YV1 <- YVP1 <- factor(rep(0,n_loops*length(Val_data[["res1"]])),levels=classes) 
  YV2 <- YVP2 <- factor(rep(0,n_loops*length(Val_data[["res2"]])),levels=classes)         #n_loops is the num of iterations, classes are the classes of Y, 
  YVP_prob1 <- data.frame(class1 = rep(0,n_loops*length(Val_data[["res1"]])),       #p are params, res is final list that will contain all the data
                          class2 = rep(0,n_loops*length(Val_data[["res1"]])))
  YVP_prob2 <- data.frame(class1 = rep(0,n_loops*length(Val_data[["res2"]])),
                          class2 = rep(0,n_loops*length(Val_data[["res2"]])))
  YT1 <- YTP1 <- factor(rep(0,n_loops*length(Te_data[["res1"]])),levels=classes)
  YT2 <- YTP2 <- factor(rep(0,n_loops*length(Te_data[["res2"]])),levels=classes)
  YTP_prob1 <- data.frame(class1 = rep(0,n_loops*length(Te_data[["res1"]])),
                          class2 = rep(0,n_loops*length(Te_data[["res1"]])))
  YTP_prob2 <- data.frame(class1 = rep(0,n_loops*length(Te_data[["res2"]])),
                          class2  = rep(0,n_loops*length(Te_data[["res2"]])))
  
  ###### Assign the values obtained from first iteration
  YV1[1:length(YVa1)] <- YVa1 
  YV2[1:length(YVa2)] <- YVa2
  YVP1[1:length(YVa1)] <- Val_data[["res1"]]
  YVP2[1:length(YVa2)] <- Val_data[["res2"]]

  YVP_prob1[["class1"]][1:length(YVa1)] <- Val_data[["res1_prob"]][[classes[[1]]]]  
  YVP_prob1[["class2"]][1:length(YVa1)] <- Val_data[["res1_prob"]][[classes[[2]]]]    
  YVP_prob2[["class1"]][1:length(YVa2)] <- Val_data[["res2_prob"]][[classes[[1]]]]  
  YVP_prob2[["class2"]][1:length(YVa2)] <- Val_data[["res2_prob"]][[classes[[2]]]]    
  
  YT1[1:length(YTe1)] <- YTe1
  YT2[1:length(YTe2)] <- YTe2
  YTP1[1:length(YTe1)] <- Te_data[["res1"]]
  YTP2[1:length(YTe2)] <- Te_data[["res2"]]
  
  YTP_prob1[["class1"]][1:length(YTe1)] <- Te_data[["res1_prob"]][[classes[[1]]]]  
  YTP_prob1[["class2"]][1:length(YTe1)] <- Te_data[["res1_prob"]][[classes[[2]]]]
  YTP_prob2[["class1"]][1:length(YTe2)] <- Te_data[["res2_prob"]][[classes[[1]]]]
  YTP_prob2[["class2"]][1:length(YTe2)] <- Te_data[["res2_prob"]][[classes[[2]]]]
  
  
  res <- append(res,list(list(params = p, YV1 = YV1,
                              YV2= YV2, YVP1 = YVP1, YVP2 = YVP2,
                              YVP_prob1 = YVP_prob1, YVP_prob2 = YVP_prob2,
                              YT1 = YT1, YT2 = YT2,
                              YTP1 = YTP1, YTP2 = YTP2,
                              YTP_prob1 = YTP_prob1, YTP_prob2 = YTP_prob2)))
  
  return(res)
}


put_data <- function(res, data, Y1, Y2, it, idx, classes){                           #put the predicted data "data" and true data "Y1","Y2 of iteration "it" 
  #inside the indexes "idx" of "res" and return "res"
  res[[idx[1]]][((it-1)*length(Y1)+1):(it*length(Y1))] <- Y1            
  res[[idx[2]]][((it-1)*length(Y2)+1):(it*length(Y2))] <- Y2
  res[[idx[3]]][((it-1)*length(Y1)+1):(it*length(Y1))] <- data[['res1']]
  res[[idx[4]]][((it-1)*length(Y2)+1):(it*length(Y2))] <- data[['res2']]
  res[[idx[5]]][["class1"]][((it-1)*length(Y1)+1):(it*length(Y1))] <- data[['res1_prob']][[classes[[1]]]] 
  res[[idx[5]]][["class2"]][((it-1)*length(Y1)+1):(it*length(Y1))] <- data[['res1_prob']][[classes[[2]]]]   
  res[[idx[6]]][["class1"]][((it-1)*length(Y2)+1):(it*length(Y2))] <- data[['res2_prob']][[classes[[1]]]]
  res[[idx[6]]][["class2"]][((it-1)*length(Y2)+1):(it*length(Y2))] <- data[['res2_prob']][[classes[[2]]]]
  
  return(res)
}


##########  Values initialization
#countries <- c("italian","chinese")
countries <- c("Philippines","Kenya")
#Y_names <- c("sugar", "salt", "butter", "peanut", "olive oil", "chicken", "veal", "pork")
Y_names <- c("funded")
n_loops <- 1        #number of times the experiment is repeated and dataset splitted
n_bstrap <- 500   #number of bootstrap samples used for avg and std of AUC
n_trees <- 1000   #number f trees random forest
n_tr_val <- 60000   #training + validation samples
p <- .9           #90% of training and 10 of validation
n_tr <- as.integer(n_tr_val * p)
n_val <- n_tr_val - as.integer(n_tr_val * p)
models <- c(countries[[1]], countries[[2]], paste(countries[[1]],"+",countries[[2]]), paste(countries[[1]],"+",countries[[2]],"+ CountLabel")) #name of the models used    
#classes <- c("Present", "Not Present")                                          #name of the classes
classes <- c("Funded","Not Funded")    
All_models_vals <- vector(mode = 'list', length = length(models))               #this list contains all the values needed for the evaluation
names(All_models_vals) <- models


#path <- "C:/Users/ariel/Documents/GitHub/Cultural-Analysis/Datasets/whats-cooking/train.json"
path <- "C:/Users/ariel/Documents/GitHub/Cultural-Analysis/Datasets/kiva_loans.csv"
setwd(path)
#data <- read.csv("food_dataset.csv", sep = ',')     
data <- read.csv("kiva_loans.csv", sep = ',') 
#all_data <- read_data_food(data, country1 = countries[[1]], country2 = countries[[2]])
all_data <- read_data_loans(data, country1 = countries[[1]], country2 = countries[[2]])

x_col=c(1:6,8:ncol(all_data[["data_country1"]]))
y_col=7

######### Tuples of parameters for each model
## SVM
C_svm <- sapply(seq(-4+7/25,3,7/25), function(x) 10^x)
gamma_svm <- sapply(seq(-4+7/25,3,7/25), function(x) 10^x)
#weigth_svm <- sapply(seq(0+1/20,1,1/20), function(x) list(c("Savory"=x,"Sweet"=1-x)))
svm_param <- list(cost = C_svm, gamma = gamma_svm)#, class.weights = weigth_svm)

## RF
mtry <- c(1, 2, 4, 5, 6, 8, 10, 12, 15, 18, 20, 24)
rf_param <- list(.mtry = mtry)


model_grid <- expand.grid(rf_param)    #create all the possible tuples of parameters


for(pred in y_col){ #for each Y
  
  for(i in c(1:n_loops)){  #repeat everything n_loops times
    
    ###### Extract the training,validation,test sets
    while(TRUE){
      data_country1 <- extract_sets(all_data[["data_country1"]][,c(pred,x_col)],n_tr,n_val) #they contain XTr, XVal, XTe, YTr, YVal, YTe
      data_country2 <- extract_sets(all_data[["data_country2"]][,c(pred,x_col)],n_tr,n_val)
      #table(data_western[['YTr']])[['Food']]
      # To avoid problems in the training phase, it must be insured that at least one sample of each class is present inside the training set
      if(table(data_country1[['YTr']])[[classes[[1]]]] != 0 && table(data_country1[['YTr']])[[classes[[2]]]] != 0 &&
         table(data_country2[['YTr']])[[classes[[1]]]] != 0 && table(data_country2[['YTr']])[[classes[[2]]]] != 0)
        break
    }
    
    ####### Iterate over all the parameters
    for(par in 1:nrow(model_grid)){
      print(paste("iteration","i:",i,"par:",par))
      for(j in 1:length(models)){   
        sets <- choose_TrValTe_sets(models[[j]], data_country1, data_country2, models)  #it contains XTr,XVal,XTe_As,XTe_We,YTr,YVal used for the specific models
        if(FALSE){
          val_results <- svm_model(sets[["XTr"]], sets[["YTr"]], sets[["XVal1"]],sets[["XVal2"]],
                                   model_grid[[par,'gamma']],model_grid[[par,'cost']])
          te_results <- svm_model(rbind(sets[["XTr"]],sets[["XVal"]]), c(as.array(sets[["YTr"]]),as.array(sets[["YVal"]])), sets[["XTe1"]], sets[["XTe2"]],
                                  model_grid[[par,'gamma']],model_grid[[par,'cost']])}
        
        val_results <- rf_model(sets[["XTr"]], sets[["YTr"]], sets[["XVal1"]],sets[["XVal2"]],
                                n_trees, model_grid[[par,'.mtry']])
        te_results <- rf_model(rbind(sets[["XTr"]],sets[["XVal"]]), c(as.array(sets[["YTr"]]),as.array(sets[["YVal"]])), sets[["XTe1"]], sets[["XTe2"]],
                               n_trees, model_grid[[par,'.mtry']])
        
        if(i==1){ 
          All_models_vals[[j]] <- init_final_data(val_results,te_results,data_country1[["YVal"]],data_country2[["YVal"]],
                                                  data_country1[["YTe"]],data_country2[["YTe"]],n_loops,
                                                  classes, model_grid[par,], All_models_vals[[j]])
        }
        else{
          All_models_vals[[j]][[par]] <- put_data(All_models_vals[[j]][[par]], val_results, data_country1[["YVal"]], data_country2[["YVal"]], it=i, idx=c(2:7),classes)
          All_models_vals[[j]][[par]] <- put_data(All_models_vals[[j]][[par]], te_results, data_country1[["YTe"]], data_country2[["YTe"]], it=i, idx=c(8:13),classes)
        }
      }
    }
  }
  print("Saving data...")
  Jdata <- toJSON(All_models_vals)
  dir.create(file.path(path,"results"), showWarnings = FALSE)
  setwd(file.path(path, "results"))
  write(Jdata, paste("Jdata_rf_",countries[[1]],"VS",countries[[2]],"_",Y_names[[1]],".json",sep = ""))
  print("Data saved...")
}
