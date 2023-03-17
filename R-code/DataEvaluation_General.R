library(grid) 
library(gridExtra)
library(pROC)
library(caret)
library(MLmetrics)
library(ggplot2)
#library(rjson)
#library(jsonlite)
library(RJSONIO)
#library(ndjson)

#rm(list=ls())
cat("\014")
set.seed(11)
graphics.off()



to_fact <- function(d,class_vect){
  for(i in c(1:length(d))){
    if(d[[i]]==0)
      d[[i]]=class_vect[0]
    else{d[[i]]=class_vect[1]}
  }
  d=factor(d)
  return(d)
}


b_acc <- function(y_true, y_pred, positive = NULL){
  y_true <- factor(y_true)
  y_pred <- factor(y_pred)
  cf <- confusionMatrix(y_pred, y_true, positive = positive)
  return(cf$byClass[['Balanced Accuracy']])
  
}

k_cohen <- function(y_true, y_pred, positive = NULL){
  y_true <- factor(y_true)
  y_pred <- factor(y_pred)
  cf <- confusionMatrix(y_pred, y_true, positive = positive)
  return(cf$overall[['Kappa']])
  
}

validation_data_roc <- function(model,data,all_models){
  
  if(model==all_models[[1]]){
    y_true = factor(data[["YV1"]])
    y_pred_c1 = data[["YVP_prob1"]]$class1  
    y_pred_c2 = data[["YVP_prob1"]]$class2   
  }
  else if(model==all_models[[2]]){
    y_true = factor(data[["YV2"]])
    y_pred_c1 = data[["YVP_prob2"]]$class1
    y_pred_c2 = data[["YVP_prob2"]]$class2
  }
  else if(model==all_models[[3]] || model==all_models[[4]]){
    y_true = factor(c(data[["YV1"]],data[["YV2"]]))
    y_pred_c1 = c(data[["YVP_prob1"]]$class1,data[["YVP_prob2"]]$class1)
    y_pred_c2 = c(data[["YVP_prob1"]]$class2,data[["YVP_prob2"]]$class2)
  }
  return(list(y_true,y_pred_c1,y_pred_c2))
}

validation_data <- function(model,data,all_models){
  
  if(model==all_models[[1]]){
    y_true = data[["YV1"]]
    y_pred = data[["YVP1"]]
  }
  else if(model==all_models[[2]]){
    y_true = data[["YV2"]]
    y_pred = data[["YVP2"]]
  }
  else if(model==all_models[[3]] || model==all_models[[4]]){
    y_true = c(data[["YV1"]],data[["YV2"]])
    y_pred = c(data[["YVP1"]],data[["YVP2"]])
  }
  return(list(y_true,y_pred))
}

saving_path <- "C:/Users/ariel/Documents/PHD/GitHub/Datasets/whats-cooking/train.json/results"
setwd(saving_path)
#countries <- c("Philippines","Kenya")
countries <- c("thai","cajun_creole")
#models <- c("italian", "chinese", "italian + chinese", "italian + chinese + CuisLabel")
models <- c(countries[[1]], countries[[2]], paste(countries[[1]],"+",countries[[2]]), paste(countries[[1]],"+",countries[[2]],"+ CountryLabel")) #name of the models used  
models_name <- c(countries[[1]], countries[[2]], paste(countries[[1]],"+",countries[[2]],sep=""), paste(countries[[1]],"+",countries[[2]],"+cuisine"),sep="") 
#models_plot <- c("italian", "thai", "italian + thai") ##it's specific of the dataset, useful for plots
classes <- c("Present", "Not Present")
#classes <- c("Funded","Not Funded")  
#d_type <- c('As','We')
n_bstrap <- 500
#Y_names <- c("sugar", "salt", "butter", "peanut", "olive oil", "chicken", "veal", "pork")
#Y_names <- c("funded")
Y_names <- c("sugar")
pred <- Y_names[[1]]


names <- apply(expand.grid(models, countries), 1, paste, collapse=".")
results <- vector("list", length(names))
names(results) <- names

scores <- list(AUC = auc,f1 = MLmetrics::F1_Score, accuracy = MLmetrics::Accuracy,
               bal_accuracy = b_acc, kappa = k_cohen, 
               spec = MLmetrics::Specificity, rec = MLmetrics::Recall,
               prec = MLmetrics::Precision) 

print("Loading the data...")
data <- fromJSON(paste("Jdata_rf_",countries[[1]],"vs",countries[[2]],"_",pred,".json",sep = ""))
print("Data loaded")

##create the outputs folder if they do not exist
rocs_path <- paste("plots/rocs/",countries[[1]],"VS",countries[[2]],sep="")
cf_path <- paste("plots/cf/",countries[[1]],"VS",countries[[2]], "/",pred,sep="")
res_path <- paste("plots/all_res/",countries[[1]],"VS",countries[[2]],"/",pred,sep="")
#dir.create(file.path(saving_path,"plots"), showWarnings = FALSE)
dir.create(file.path(saving_path, rocs_path), showWarnings = FALSE, recursive=TRUE)  
dir.create(file.path(saving_path, cf_path), showWarnings = FALSE, recursive=TRUE)  
dir.create(file.path(saving_path, res_path), showWarnings = FALSE, recursive=TRUE)  


for(s in 1:length(scores)){
  country1_plots <- list()       #init of plots
  country2_plots <- list()
  country1_rocs <- list()
  country2_rocs <- list()
  it <- 1
  for(m in models){
    for(type in c(1:2)){    # western or asian models
      all_res <- list("score"=character(),"param"=list(),"v_res"=vector(),"t_res"=vector())
      max_score <- 0        # init the best score to 0
      for(p in 1:length(data[[m]])){    #for all the tuples of parameters
        #print(p)
        if(names(scores)[s]=='AUC'){
          val_res <- validation_data_roc(m,data[[m]][[p]],models)
          res <- scores[[s]](val_res[[1]],
                             val_res[[2]],
                             levels = classes, direction = ">")  
          #res <- scores[[s]](factor(data[[m]][[p]][[paste('YV_',type,sep = "")]]),
          #                   data[[m]][[p]][[paste('YVP_prob_',type,sep = "")]]$Savory,
          #                   levels = c("Savory", "Sweet"), direction = ">")
          res_test <- scores[[s]](factor(data[[m]][[p]][[paste('YT',type,sep = "")]]),
                                  data[[m]][[p]][[paste('YTP_prob',type,sep = "")]]$class1,  #####Savory
                                  levels = classes, direction = ">")
          res = res[1]  #take auc score
          res_test = res_test[1]
        }
        else{
          val_res <- validation_data(m,data[[m]][[p]],models)
          res <- scores[[s]](y_true=val_res[[1]],
                             y_pred=val_res[[2]])
          #res <- scores[[s]](y_true=data[[m]][[p]][[paste('YV_',type,sep = "")]],
          #                   y_pred=data[[m]][[p]][[paste('YVP_',type,sep = "")]])
          res_test <- scores[[s]](y_true=data[[m]][[p]][[paste('YT',type,sep = "")]],
                                  y_pred=data[[m]][[p]][[paste('YTP',type,sep = "")]])
          
        }
        all_res[["v_res"]] <- c(all_res[["v_res"]],res)
        all_res[["t_res"]] <- c(all_res[["t_res"]],res_test)
        all_res[["param"]] <- append(all_res[["param"]],list(data[[m]][[p]][["params"]]))
        if(is.na(res))   #in this case it is not possible to check res_test > max_score
          next
        if(res > max_score){     #if I found that the actual score is better than the current best_score, replace best_score with actual score
          max_score <- res
          param_best <- data[[m]][[p]][["params"]]
          p_ind <- p
        }
      }
      
      all_res[["score"]] <- names(scores)[s]
      results[[paste(m,countries[type],sep=".")]] <- all_res  #to save results of each model as "model.test", eg. "country1.italian"
      #that means model "Western" tested with "Asian" data
      if(names(scores)[s]=='AUC'){        #Compute avg and std AUC by using bootstrap samples
        bootstrap_scores = list()
        n_skip <- 0 
        for(bn in 1:n_bstrap){
          indices <- sample(length(data[[m]][[p_ind]][[paste('YT',type,sep = "")]]),##YT
                            length(data[[m]][[p_ind]][[paste('YT',type,sep = "")]]), replace = TRUE)##YT, anche sotto
          if(length(unique(data[[m]][[p_ind]][[paste('YT',type,sep = "")]][indices])) == 1){   #if there is only one class, skip and do not compute AUC
            n_skip <- n_skip + 1
            next}
          score <- auc(roc(factor(data[[m]][[p_ind]][[paste('YT',type,sep = "")]][indices]),##YT
                           data[[m]][[p_ind]][[paste('YTP_prob',type,sep = "")]]$class1[indices],##YTP_prob  ##Savory
                           levels = classes, direction = ">"))#,percent=TRUE))
          bootstrap_scores <- append(bootstrap_scores,score)
        }
        avg_score <- round(mean(unlist(bootstrap_scores)), digits = 3)
        std_score <- round(sd(unlist(bootstrap_scores)), digits = 3)
        
        
        r <- roc(factor(data[[m]][[p_ind]][[paste('YT',type,sep = "")]]), ##YT
                 data[[m]][[p_ind]][[paste('YTP_prob',type,sep = "")]]$class1,percent=TRUE,##YTP_prob  ###Savory
                 levels = classes, direction = ">")
        ac <- paste("AUC=", round(auc(r)/100, digits=3),"\n", #% \n",
                    "Avg_AUC=", avg_score, "\n", #"% \n",
                    "Std_AUC=", std_score, "\n", sep='') #"% \n", sep='')
        r_pl <- eval(substitute(ggroc(r, colour = "#377eb8", size = 1)
                                + theme_bw()
                                + geom_abline(intercept = 100, slope = 1, colour = "green", size = 0.4, linetype = 2)
                                + annotate(geom="text", x=18, y=10, label=ac, colour = "blue")
                                + ggtitle(paste("ROC of", models_name[[match(m,models)]],"model \n tested with", countries[type],"data"))  ###m where there is match
                                #+ xlab="False Positive Percentage" + ylab="True Postive Percentage"
                                + theme(panel.background = element_blank(),
                                        axis.line = element_line(colour = "black"),
                                        plot.title = element_text(hjust = 0.5, lineheight=.8, colour = 'red')),list(it = it)))
        if(type==1)
          country1_rocs[[match(m,models)]] <- r_pl
        else
          country2_rocs[[match(m,models)]] <- r_pl
        
      }
      else{
        CM <- confusionMatrix(factor(data[[m]][[p_ind]][[paste('YTP',type,sep = "")]]),  ##YTP
                              factor(data[[m]][[p_ind]][[paste('YT',type,sep = "")]]),   ##YT
                              positive = classes[[1]])  ####Savory
        cm <- as.data.frame(CM$table) #results
        cm$Freq = c(array(c(as.matrix(CM)),dim=c(2,2)))
        cm$Perc <- round((cm$Freq/sum(cm$Freq))*100,2) 
        
        cm_plot <- eval(substitute(ggplot(data = cm, aes(Prediction, Reference, fill = Freq)) +
                                     geom_tile() +
                                     #labs(x = "Reference",y = "Prediction") +
                                     scale_fill_gradient(low="white", high="#009194") +
                                     geom_text(aes(label = paste(Perc,"%","\n",
                                                                 "Freq:", Freq)), color = 'black', size = 4) +
                                     theme_light() +
                                     scale_y_discrete(limits=rev) +
                                     scale_x_discrete(position = "top") +
                                     guides(fill="none") +
                                     ggtitle(paste("CM of",models_name[[match(m,models)]],"model \n tested with", countries[type],"data")) +
                                     theme(plot.title = element_text(hjust = 0.5, lineheight=.8, colour = 'red')),list(it = it)))
        
        if(type==1)
          country1_plots[[match(m,models)]] <- cm_plot
        else
          country2_plots[[match(m,models)]] <- cm_plot
        
      } 
      it <- it + 1
    }
  }
  if(names(scores)[s]!='AUC'){
    all_cm_plot <- grid.arrange(grobs = c(country1_plots,country2_plots), nrow=2, ncol=length(models),
                                top=textGrob(paste("Confusion Matrices_",pred,sep=""),gp=gpar(fontsize=26,color = "steelblue",font=1)))
    #print(all_cm_plot)
    ggsave(paste("conf_matrices_rf_ValTe_",pred,"_",countries[[1]],"VS",countries[[2]],"_",names(scores)[s],".png",sep=""), plot = all_cm_plot, device = png(width=300*length(models), height = 600), 
           path = file.path(saving_path, cf_path))
  }
  else{
    all_roc_plot <- grid.arrange(grobs = c(country1_rocs,country2_rocs), nrow=2, ncol=length(models),
                                 top=textGrob(paste("ROCs_",pred),gp=gpar(fontsize=26,colour = "steelblue",font=1)))
    
    ggsave(paste("rocs_rf_ValTe_",pred,"_",countries[[1]],"VS",countries[[2]],".png",sep=""), plot = all_roc_plot, device = png(width=300*length(models), height = 600), 
           path = file.path(saving_path, rocs_path))#, paste(countries[[1]],"VS",countries[[2]],sep="")))
  }
  Jdata <- toJSON(results)
  write(Jdata, paste(file.path(saving_path, res_path),"/",names(scores)[s],"_rf_ValTe_",pred,"_",countries[[1]],"VS",countries[[2]],".json",sep=""))
}