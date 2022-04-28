library(dplyr)
library(randomForest)
library(caret)
library(ggplot2)
library(grid) 
library(gridExtra)
library(pROC)
library(MLmetrics)
library(RJSONIO)
library(e1071)

###### Workspace initialization
rm(list=ls())
cat("\014")
set.seed(11)
graphics.off()    #used to avoid problems during plots




min_max_norm <- function(x, na.rm = FALSE){   #na values are already removed

  di <- max(x) - min(x)
  if(di > 1e-6)                               #normalize only if difference between max and min is > 1e-6 to avoid numerical issues
    return((x- min(x)) /(max(x)-min(x)))
  else
    return(0)
}

normalization <- function(data, norm_type){
  
  for(i in 3:ncol(data)){                   #normalize all the columns (in this case there are 3 columns to be normalized)
    data[,i] <- norm_type(data[,i])
  }
  return(data)
}

read_data <- function(data, norm_fun, norm_type){
  
  options(max.print=1000000)  #print all the data
  data <- data.frame(data["Food.vs.nonfood"],data["Food.classification"],data[,9:30])        #columns from 9 to 30 contain the values used for training #data["Swet.vs.Savory"]
  #data <- filter(data, Sweet.vs.savory == "Sweet" | Sweet.vs.savory == "Savory")                #take only the foods that are sweet or savory
  data <- filter(data, Food.vs.nonfood == "Food" | Food.vs.nonfood == "Nonfood")                #take only the foods that are sweet or savory
  data <- na.omit(data)                                                                         #remove all data with empty cells
  n_nondata <- round((length(filter(data, Food.vs.nonfood == "Nonfood")[[1]])/2),0)
  nondata <- filter(data, Food.vs.nonfood == "Nonfood")
  #data$Sweet.vs.savory <- factor(data$Sweet.vs.savory)                                          #from string to factor variables
  data$Food.vs.nonfood <- factor(data$Food.vs.nonfood)                                          #from string to factor variables
  data <- norm_fun(data,norm_type)                                                              #normalize all the data so that it can be used by the models
  Data_Asian <- select(rbind((filter(data, Food.classification == "Asian")),nondata[1:n_nondata,]),-Food.classification)     #take all the asian data and remove Food Classification column
  Data_Western <- select(rbind((filter(data, Food.classification == "Western")),nondata[n_nondata+1:length(nondata),]),-Food.classification) #take all the western data and remove Food Classification column
  print(Data_Asian)
  #print(Data_Western[["Food.vs.nonfood"]])

  return(list("data"=data,"Data_Asian"=Data_Asian,"Data_Western"=Data_Western))
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


choose_TrValTe_sets <- function(model, data_we, data_as){   #depending on the model used, we will have different training and validation sets
  
  if(model=="Western"){
    XTr <- data_we[["XTr"]]
    XVal <- data_we[["XVal"]]
    YTr <- data_we[["YTr"]]
    YVal <- data_we[["YVal"]]
  }
  else if(model=="Asian"){
    XTr <- data_as[["XTr"]]
    XVal <- data_as[["XVal"]]
    YTr <- data_as[["YTr"]]
    YVal <- data_as[["YVal"]]
  }
  else if(model=="Western + Asian"){
    XTr <- rbind(data_we[["XTr"]], data_as[["XTr"]])
    XVal <- rbind(data_we[["XVal"]], data_as[["XVal"]])
    YTr <- c(as.array(data_we[["YTr"]]), as.array(data_as[["YTr"]]))
    YVal <- c(as.array(data_we[["YVal"]]), as.array(data_as[["YVal"]]))
  }
  else if(model=="Western + Asian + FC labels"){    
    #Add a column of 1 if food is asian, 1 if it is western
    data_as[["XTr"]] <- cbind(data_as[["XTr"]], Food_Classification = rep(0,nrow(data_as[["XTr"]])))
    data_as[["XVal"]] <- cbind(data_as[["XVal"]], Food_Classification = rep(0,nrow(data_as[["XVal"]])))
    data_as[["XTe"]] <- cbind(data_as[["XTe"]], Food_Classification = rep(0,nrow(data_as[["XTe"]])))
    data_we[["XTr"]] <- cbind(data_we[["XTr"]], Food_Classification = rep(1,nrow(data_we[["XTr"]])))
    data_we[["XVal"]] <- cbind(data_we[["XVal"]], Food_Classification = rep(1,nrow(data_we[["XVal"]])))
    data_we[["XTe"]] <- cbind(data_we[["XTe"]], Food_Classification = rep(1,nrow(data_we[["XTe"]])))

    XTr <- rbind(data_we[["XTr"]], data_as[["XTr"]])
    XVal <- rbind(data_we[["XVal"]], data_as[["XVal"]])
    YTr <- c(as.array(data_we[["YTr"]]), as.array(data_as[["YTr"]]))
    YVal <- c(as.array(data_we[["YVal"]]), as.array(data_as[["YVal"]]))
  }
  return(list("XTr"=XTr,"XVal"=XVal,"YTr"=YTr,"YVal"=YVal,"XTe_As"=data_as[["XTe"]],"XTe_We"=data_we[["XTe"]],
              "XVal_As"=data_as[["XVal"]],"XVal_We"=data_we[["XVal"]]))
}


rf_model <- function(xtr, ytr, xte_we, xte_as, n_trees, mtry){
  
  M <- randomForest(x = xtr, y = ytr, importance = TRUE, ntree = n_trees,       #build the random forest model and train it using the tr_data and tuple of params
                    sampsize = c(min(summary(ytr)),min(summary(ytr))),
                    mtry = mtry)
  
  #compute predictions and probabilities
  Y_Asian <- predict(M,xte_as)
  Y_Western <- predict(M,xte_we)
  Y_Asian_prob <- as.data.frame(predict(M,xte_as, type = "prob")) 
  Y_Western_prob <- as.data.frame(predict(M,xte_we, type = "prob")) 
  
  return(list("res_as"=Y_Asian,"res_we"=Y_Western,
              "res_as_prob"=Y_Asian_prob, "res_we_prob"=Y_Western_prob))
}


svm_model <- function(xtr,ytr,xte_we,xte_as,gamma,cost){
  
  unb_fact <- table(c(ytr))[['Food']]/table(c(ytr))[['Nonfood']]                #compute unbalance between classes   ######Savory then Sweet
  M <- svm(x = xtr, y = ytr, kernel = "radial",                                 #build the svm model and train it using the tr_data and tuple of params
           gamma = gamma, cost = cost,
           probability=TRUE, class.weights = c("Food"=1/(1+unb_fact),"Nonfood"=1-(1/(1+unb_fact))))    ######Savory then Sweet
  #compute predictions and probabilities
  Y_Asian <- predict(M,xte_as)
  Y_Western <- predict(M,xte_we)
  Y_Asian_prob <- as.data.frame(attr(predict(M,xte_as,probability=TRUE),"probabilities")) 
  Y_Western_prob <- as.data.frame(attr(predict(M,xte_we,probability=TRUE),"probabilities")) 
  
  return(list("res_as"=Y_Asian,"res_we"=Y_Western,
              "res_as_prob"=Y_Asian_prob, "res_we_prob"=Y_Western_prob))
}

init_final_data <- function(Val_data,Te_data,YVa_We,YVa_As,YTe_We,YTe_As,n_loops,classes,p,res){  #Val_data contains Y predicted by using validation sets
  ##########Factors were of Savory then Sweet
  ##### Vectors initialization                                                                #Te_data contains Y predicted by using test sets
  YV_We <- YVP_We <- factor(rep(0,n_loops*length(Val_data[["res_we"]])),levels=classes) #YV_We,YV_As,YT_We,YT_As contain real data used for validation and test
  YV_As <- YVP_As <- factor(rep(0,n_loops*length(Val_data[["res_as"]])),levels=classes) #n_loops is the # of iterations, classes are the classes of Y, p are params, res is final list
  YVP_prob_We <- data.frame(Food = rep(0,n_loops*length(Val_data[["res_we"]])),
                            Nonfood = rep(0,n_loops*length(Val_data[["res_we"]])))
  YVP_prob_As <- data.frame(Food = rep(0,n_loops*length(Val_data[["res_as"]])),
                            Nonfood = rep(0,n_loops*length(Val_data[["res_as"]])))
  YT_We <- YTP_We <- factor(rep(0,n_loops*length(Te_data[["res_we"]])),levels=classes)
  YT_As <- YTP_As <- factor(rep(0,n_loops*length(Te_data[["res_as"]])),levels=classes)
  YTP_prob_We <- data.frame(Food = rep(0,n_loops*length(Te_data[["res_we"]])),
                            Nonfood = rep(0,n_loops*length(Te_data[["res_we"]])))
  YTP_prob_As <- data.frame(Food = rep(0,n_loops*length(Te_data[["res_as"]])),
                            Nonfood = rep(0,n_loops*length(Te_data[["res_as"]])))
  
  ###### Assign the values obtained from first iteration
  YV_We[1:length(YVa_We)] <- YVa_We 
  YV_As[1:length(YVa_As)] <- YVa_As
  YVP_We[1:length(YVa_We)] <- Val_data[["res_we"]]
  YVP_As[1:length(YVa_As)] <- Val_data[["res_as"]]
  
  YVP_prob_We[['Food']][1:length(YVa_We)] <- Val_data[["res_we_prob"]][['Food']]  #####Savory
  YVP_prob_We[['Nonfood']][1:length(YVa_We)] <- Val_data[["res_we_prob"]][['Nonfood']]    #####Sweet
  YVP_prob_As[['Food']][1:length(YVa_As)] <- Val_data[["res_as_prob"]][['Food']]  #####Savory
  YVP_prob_As[['Nonfood']][1:length(YVa_As)] <- Val_data[["res_as_prob"]][['Nonfood']]    #####Sweet
  
  YT_We[1:length(YTe_We)] <- YTe_We
  YT_As[1:length(YTe_As)] <- YTe_As
  YTP_We[1:length(YTe_We)] <- Te_data[["res_we"]]
  YTP_As[1:length(YTe_As)] <- Te_data[["res_as"]]
  
  YTP_prob_We[['Food']][1:length(YTe_We)] <- Te_data[["res_we_prob"]][['Food']]  #####As before, same order
  YTP_prob_We[['Nonfood']][1:length(YTe_We)] <- Te_data[["res_we_prob"]][['Nonfood']]
  YTP_prob_As[['Food']][1:length(YTe_As)] <- Te_data[["res_as_prob"]][['Food']]
  YTP_prob_As[['Nonfood']][1:length(YTe_As)] <- Te_data[["res_as_prob"]][['Nonfood']]
  
  
  res <- append(res,list(list(params = p, YV_As = YV_As,
                              YV_We = YV_We, YVP_As = YVP_As, YVP_We = YVP_We,
                              YVP_prob_As = YVP_prob_As, YVP_prob_We = YVP_prob_We,
                              YT_As = YT_As, YT_We = YT_We,
                              YTP_As = YTP_As, YTP_We = YTP_We,
                              YTP_prob_As = YTP_prob_As, YTP_prob_We = YTP_prob_We)))
  
  return(res)
}

put_data <- function(res, data, Y_We, Y_As, it, idx){                           #put the predicted data "data" and true data "Y_We","Y_As of iteration "it" 
                                                                                #inside the indexes "idx" of "res" and return "res"
  res[[idx[1]]][((it-1)*length(Y_As)+1):(it*length(Y_As))] <- Y_As            
  res[[idx[2]]][((it-1)*length(Y_We)+1):(it*length(Y_We))] <- Y_We
  res[[idx[3]]][((it-1)*length(Y_As)+1):(it*length(Y_As))] <- data[['res_as']]
  res[[idx[4]]][((it-1)*length(Y_We)+1):(it*length(Y_We))] <- data[['res_we']]
  res[[idx[5]]][['Food']][((it-1)*length(Y_As)+1):(it*length(Y_As))] <- data[['res_as_prob']][['Food']] ###Transformed
  res[[idx[5]]][['Nonfood']][((it-1)*length(Y_As)+1):(it*length(Y_As))] <- data[['res_as_prob']][['Nonfood']]   ###Natural
  res[[idx[6]]][['Food']][((it-1)*length(Y_We)+1):(it*length(Y_We))] <- data[['res_we_prob']][['Food']]
  res[[idx[6]]][['Nonfood']][((it-1)*length(Y_We)+1):(it*length(Y_We))] <- data[['res_we_prob']][['Nonfood']]
  
  return(res)
}

print_sizes <- function(data){
  
    #print(object.size(All_models_vals))
    print(paste("YV_We:",object.size(data[['YV_We']]),
                " YV_As:", object.size(data[['YV_As']]),
                " YVP_As",object.size(data[['YVP_As']]),
                " YVP_We",object.size(data[['YVP_We']]),
                " YVP_prob_We:",object.size(data[['YVP_prob_We']]),
                " YVP_prob_As:",object.size(data[['YVP_prob_As']]),
                " YT_We:",object.size(data[['YT_We']]),
                " YT_As:",object.size(data[['YT_As']]),
                " YTP_We:",object.size(data[['YTP_We']]),
                " YTP_As:",object.size(data[['YTP_As']]),
                " YTP_prob_We",object.size(data[['YTP_prob_We']]),
                " YTP_prob_As",object.size(data[['YTP_prob_As']])))
}



######### Data loading
setwd("C:/users/ariel/Documents/PHD/osfstorage-archive/code/randomforest")
data <- read.csv("All_Foodpictures_information_132.csv", sep = ';')####before there was not 2 and no sep
d <- read_data(data, norm_fun = normalization, norm_type = min_max_norm)
#print(d[["Data_Western"]])



##########  Values initialization
n_loops <- 30        #number of times the experiment is repeated and dataset splitted
n_bstrap <- 500   #number of bootstrap samples used for avg and std of AUC
n_trees <- 1000   #number f trees random forest
n_tr_val <- 190   #training + validation samples
p <- .9           #90% of training and 10 of validation
n_tr <- as.integer(n_tr_val * p)
n_tr_as <- 10
n_val <- n_tr_val - as.integer(n_tr_val * p)
models <- c("Western", "Asian", "Western + Asian","Western + Asian + FC labels")  #name of the models used
tests <- c("Asian", "Western")                                                    #name of the tests data used on models
#classes <- c("Sweet", "Savory")                                                   #name of the classes
classes <- c("Nonfood", "Food")  
All_models_vals <- vector(mode = 'list', length = length(models))                 #this list contains all the values needed for the evaluation
names(All_models_vals) <- models


######### Tuples of parameters for each model

## SVM
C_svm <- sapply(seq(-4+7/25,3,7/25), function(x) 10^x)
gamma_svm <- sapply(seq(-4+7/25,3,7/25), function(x) 10^x)
#weigth_svm <- sapply(seq(0+1/20,1,1/20), function(x) list(c("Savory"=x,"Sweet"=1-x)))
svm_param <- list(cost = C_svm, gamma = gamma_svm)#, class.weights = weigth_svm)

## RF
mtry <- c(1, 2, 4, 5, 6, 8, 10)
rf_param <- list(.mtry = mtry)


model_grid <- expand.grid(rf_param)    #create all the possible tuples of parameters


for(i in c(1:n_loops)){  #repeat everything 30 times

  ###### Extract the training,validation,test sets
  while(TRUE){
    data_western <- extract_sets(d[["Data_Western"]],n_tr,n_val) #they contain XTr, XVal, XTe, YTr, YVal, YTe
    data_asian <- extract_sets(d[["Data_Asian"]],n_tr,n_val)
    table(data_western[['YTr']])[['Food']]
    # To avoid problems in the training phase, it must be insured that at least one sample of each class is present inside the training set
    if(table(data_western[['YTr']])[['Food']] != 0 && table(data_western[['YTr']])[['Nonfood']] != 0 &&
       table(data_asian[['YTr']])[['Food']] != 0 && table(data_asian[['YTr']])[['Nonfood']] != 0)
      break
  }

  ####### Iterate over all the parameters
  for(par in 1:nrow(model_grid)){
    print(paste("iteration","i:",i,"par:",par))
    for(j in 1:4){   
      sets <- choose_TrValTe_sets(models[[j]], data_western, data_asian)  #it contains XTr,XVal,XTe_As,XTe_We,YTr,YVal used for the specific models
      if(FALSE){
      val_results <- svm_model(sets[["XTr"]], sets[["YTr"]], sets[["XVal_We"]],sets[["XVal_As"]],
                               model_grid[[par,'gamma']],model_grid[[par,'cost']])
      te_results <- svm_model(rbind(sets[["XTr"]],sets[["XVal"]]), c(as.array(sets[["YTr"]]),as.array(sets[["YVal"]])), sets[["XTe_We"]], sets[["XTe_As"]],
                               model_grid[[par,'gamma']],model_grid[[par,'cost']])}
    
      val_results <- rf_model(sets[["XTr"]], sets[["YTr"]], sets[["XVal_We"]],sets[["XVal_As"]],
                              n_trees, model_grid[[par,'.mtry']])
      te_results <- rf_model(rbind(sets[["XTr"]],sets[["XVal"]]), c(as.array(sets[["YTr"]]),as.array(sets[["YVal"]])), sets[["XTe_We"]], sets[["XTe_As"]],
                             n_trees, model_grid[[par,'.mtry']])
      
      if(i==1){ 
        All_models_vals[[j]] <- init_final_data(val_results,te_results,data_western[["YVal"]],data_asian[["YVal"]],
                                                data_western[["YTe"]],data_asian[["YTe"]],n_loops,
                                                classes, model_grid[par,], All_models_vals[[j]])
      }
      else{
        All_models_vals[[j]][[par]] <- put_data(All_models_vals[[j]][[par]], val_results, data_western[["YVal"]], data_asian[["YVal"]], it=i, idx=c(2:7))
        All_models_vals[[j]][[par]] <- put_data(All_models_vals[[j]][[par]], te_results, data_western[["YTe"]], data_asian[["YTe"]], it=i, idx=c(8:13))
      }
    }
  }
}
print("Saving data...")
Jdata <- toJSON(All_models_vals)
write(Jdata, "Jdata_rf_FvsNF.json")
print("Data saved...")