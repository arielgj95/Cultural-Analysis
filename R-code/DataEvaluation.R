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

validation_data_roc <- function(model,data){

  if(model=="Western"){
    y_true = factor(data[["YV_We"]])
    y_pred_sav = data[["YVP_prob_We"]]$Food  ####Transformed=Savory
    y_pred_swe = data[["YVP_prob_We"]]$Nonfood    ####Natural=Sweet
  }
  else if(model=="Asian"){
    y_true = factor(data[["YV_As"]])
    y_pred_sav = data[["YVP_prob_As"]]$Food
    y_pred_swe = data[["YVP_prob_As"]]$Nonfood
  }
  else if(model=="Western + Asian" || model=="Western + Asian + FC labels"){
    y_true = factor(c(data[["YV_We"]],data[["YV_As"]]))
    y_pred_sav = c(data[["YVP_prob_We"]]$Food,data[["YVP_prob_As"]]$Food)
    y_pred_swe = c(data[["YVP_prob_We"]]$Nonfood,data[["YVP_prob_As"]]$Nonfood)
  }
  return(list(y_true,y_pred_sav,y_pred_swe))
}

validation_data <- function(model,data){
  
  if(model=="Western"){
    y_true = data[["YV_We"]]
    y_pred = data[["YVP_We"]]
  }
  else if(model=="Asian"){
    y_true = data[["YV_As"]]
    y_pred = data[["YVP_As"]]
  }
  else if(model=="Western + Asian" || model=="Western + Asian + FC labels"){
    y_true = c(data[["YV_We"]],data[["YV_As"]])
    y_pred = c(data[["YVP_We"]],data[["YVP_As"]])
  }
  return(list(y_true,y_pred))
}

setwd("C:/users/ariel/Documents/PHD/osfstorage-archive/code/randomforest")
models <- c("Western", "Asian", "Western + Asian","Western + Asian + FC labels")
tests <- c("Asian", "Western")
#classes <- c("Savory", "Sweet")
#classes <- c("Transformed", "Natural")
classes <- c("Food", "Nonfood")
d_type <- c('As','We')
n_bstrap <- 500

names <- apply(expand.grid(models, tests), 1, paste, collapse=".")
results <- vector("list", length(names))
names(results) <- names

scores <- list(AUC = auc,f1 = MLmetrics::F1_Score, accuracy = MLmetrics::Accuracy,
               bal_accuracy = b_acc, kappa = k_cohen, 
               spec = MLmetrics::Specificity, rec = MLmetrics::Recall,
               prec = MLmetrics::Precision) 

print("Loading the data...")
data <- fromJSON("Jdata_rf_FvsNF.json")
print("Data loaded")
#val_score

for(s in 1:length(scores)){
  asian_plots <- list()       #init of plots
  western_plots <- list()
  asian_rocs <- list()
  western_rocs <- list()
  it <- 1
  for(m in models){
    for(type in d_type){    # western or asian models
      all_res <- list("score"=character(),"param"=list(),"v_res"=vector(),"t_res"=vector())
      max_score <- 0        # init the best score to 0
      for(p in 1:length(data[[m]])){    #for all the tuples of parameters
        #print(p)
        if(names(scores)[s]=='AUC'){
          val_res <- validation_data_roc(m,data[[m]][[p]])
          res <- scores[[s]](val_res[[1]],
                             val_res[[2]],
                             levels = classes, direction = ">")  
          #res <- scores[[s]](factor(data[[m]][[p]][[paste('YV_',type,sep = "")]]),
          #                   data[[m]][[p]][[paste('YVP_prob_',type,sep = "")]]$Savory,
          #                   levels = c("Savory", "Sweet"), direction = ">")
          res_test <- scores[[s]](factor(data[[m]][[p]][[paste('YT_',type,sep = "")]]),
                             data[[m]][[p]][[paste('YTP_prob_',type,sep = "")]]$Food,  #####Savory
                             levels = classes, direction = ">")
          res = res[1]  #take auc score
          res_test = res_test[1]
        }
        else{
          val_res <- validation_data(m,data[[m]][[p]])
          res <- scores[[s]](y_true=val_res[[1]],
                             y_pred=val_res[[2]])
          #res <- scores[[s]](y_true=data[[m]][[p]][[paste('YV_',type,sep = "")]],
          #                   y_pred=data[[m]][[p]][[paste('YVP_',type,sep = "")]])
          res_test <- scores[[s]](y_true=data[[m]][[p]][[paste('YT_',type,sep = "")]],
                             y_pred=data[[m]][[p]][[paste('YTP_',type,sep = "")]])

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
      results[[paste(m,tests[match(type,d_type)],sep=".")]] <- all_res  #to save results of each model as "model.test", eg. "Western.Asian"
                                                                        #that means model "Western" tested with "Asian" data
      if(names(scores)[s]=='AUC'){        #Compute avg and std AUC by using bootstrap samples
        bootstrap_scores = list()
        n_skip <- 0 
        for(bn in 1:n_bstrap){
          indices <- sample(length(data[[m]][[p_ind]][[paste('YT_',type,sep = "")]]),##YT
                            length(data[[m]][[p_ind]][[paste('YT_',type,sep = "")]]), replace = TRUE)##YT, anche sotto
          if(length(unique(data[[m]][[p_ind]][[paste('YT_',type,sep = "")]][indices])) == 1){   #if there is only one class, skip and do not compute AUC
            n_skip <- n_skip + 1
            next}
          score <- auc(roc(factor(data[[m]][[p_ind]][[paste('YT_',type,sep = "")]][indices]),##YT
                           data[[m]][[p_ind]][[paste('YTP_prob_',type,sep = "")]]$Food[indices],##YTP_prob  ##Savory
                           levels = classes, direction = ">"))#,percent=TRUE))
          bootstrap_scores <- append(bootstrap_scores,score)
        }
        avg_score <- round(mean(unlist(bootstrap_scores)), digits = 3)
        std_score <- round(sd(unlist(bootstrap_scores)), digits = 3)
        
        
        r <- roc(factor(data[[m]][[p_ind]][[paste('YT_',type,sep = "")]]), ##YT
                 data[[m]][[p_ind]][[paste('YTP_prob_',type,sep = "")]]$Food,percent=TRUE,##YTP_prob  ###Savory
                 levels = classes, direction = ">")
        ac <- paste("AUC=", round(auc(r)/100, digits=3),"\n", #% \n",
                    "Avg_AUC=", avg_score, "\n", #"% \n",
                    "Std_AUC=", std_score, "\n", sep='') #"% \n", sep='')
        r_pl <- eval(substitute(ggroc(r, colour = "#377eb8", size = 1)
                                + theme_bw()
                                + geom_abline(intercept = 100, slope = 1, colour = "green", size = 0.4, linetype = 2)
                                + annotate(geom="text", x=18, y=10, label=ac, colour = "blue")
                                + ggtitle(paste("ROC of", m,"model \n tested with", tests[match(type,d_type)],"data"))
                                #+ xlab="False Positive Percentage" + ylab="True Postive Percentage"
                                + theme(panel.background = element_blank(),
                                        axis.line = element_line(colour = "black"),
                                        plot.title = element_text(hjust = 0.5, lineheight=.8, colour = 'red')),list(it = it)))
        if(type=='As')
          asian_rocs[[match(m,models)]] <- r_pl
        else
          western_rocs[[match(m,models)]] <- r_pl
        
      }
      else{
        CM <- confusionMatrix(factor(data[[m]][[p_ind]][[paste('YTP_',type,sep = "")]]),  ##YTP
                              factor(data[[m]][[p_ind]][[paste('YT_',type,sep = "")]]),   ##YT
                              positive = "Food")  ####Savory
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
                                     ggtitle(paste("CM of", m,"model \n tested with", tests[match(type,d_type)],"data")) +
                                     theme(plot.title = element_text(hjust = 0.5, lineheight=.8, colour = 'red')),list(it = it)))
        
        if(type=='As')
          asian_plots[[match(m,models)]] <- cm_plot
        else
          western_plots[[match(m,models)]] <- cm_plot
        
      } 
      it <- it + 1
    }
  }
  if(names(scores)[s]!='AUC'){
    all_cm_plot <- grid.arrange(grobs = c(asian_plots,western_plots), nrow=2, ncol=4,
                                top=textGrob("Confusion Matrices SvsS",gp=gpar(fontsize=26,color = "steelblue",font=1)))
    #print(all_cm_plot)
    ggsave(paste("conf_matrices_rf_ValTe_FvsNF_",names(scores)[s],".png",sep=""), plot = all_cm_plot, device = png(width=1200, height = 600), 
         path = "C:/Users/ariel/Documents/PHD/osfstorage-archive/code/plot/CM")
  }
  else{
  all_roc_plot <- grid.arrange(grobs = c(asian_rocs,western_rocs), nrow=2, ncol=4,
                               top=textGrob("ROCs SvsS",gp=gpar(fontsize=26,colour = "steelblue",font=1)))
  
  ggsave(paste("rocs_rf_ValTe_FvsNF.png"), plot = all_roc_plot, device = png(width=1200, height = 600), 
         path = "C:/Users/ariel/Documents/PHD/osfstorage-archive/code/plot/rocs")
  }
  Jdata <- toJSON(results)
  write(Jdata, paste("C:/Users/ariel/Documents/PHD/osfstorage-archive/code/plot/all_res_",names(scores)[s],"_rf_ValTe_FvsNF.json",sep=""))
}