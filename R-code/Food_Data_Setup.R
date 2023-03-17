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

X_names <- c("tomato", "corn", "rice", "broth", "beans", "carrots", "avocado",
             "beef", "bread", "potatoes", "shrimp", "tortillas", "fish", "meat",
             "mushrooms", "peas", "noodles", "spinach", "olives", "radishes",
             "eel", "cabbage", "lettuce", "bacon", "mozzarella cheese", "pasta",
             "cucumber", "steak", "loin", "zucchini", "ham", "toast", "sirloin",
             "turkey", "lamb", "chips", "broccoli", "spaghetti", "kalamata",
             "pineapple", "asparagus", "mango", "baguette", "salad", "stew",
             "penne", "cauliflower", "salmon", "lentils", "basmati","olive oil", 
             "salt","pepper","yeast","flour","onions","vinegar","butter")

Y_names <- c("sugar", "salt", "butter", "peanut", "olive oil", "chicken", "veal", "pork")


transform_data <- function(data){
  
  options(max.print=1000000)  #print all the data
  n_col <- length(c(X_names,Y_names))+1
  final_data = data.frame(matrix(nrow=0, ncol = n_col )) 
  for(d in 1:(length(data))){                                                          # for all the data
    row_element = rep(0,n_col)                                                            #init row element
    row_element[[n_col]] <- data[[d]][['cuisine']]
    for(r in 1:(n_col-1)){                                                           # for each X,Y
      for(ing in data[[d]][['ingredients']]){                                           #for all the ingredients of a recipe
        if(grepl(c(X_names,Y_names)[[r]], ing, fixed = TRUE)){                   #check if the ingredient X is inside the ingredients of the recipe j (it is sufficient that the)
          row_element[[r]] <- 1                                                                           #ingredient X string is inside one of the ingredient j strings)
          break
        }
      }
    }
    final_data <- rbind(final_data,row_element)
  }
  colnames(final_data) = c(X_names,Y_names,"cuisine")
  return(final_data)
}


## Data loading
setwd("C:/Users/ariel/Documents/GitHub/Datasets/whats-cooking/train.json")
data <- fromJSON("train.json")
## Obtaining and saving new data
final_data <- transform_data(data)
write.csv(final_data,"food_dataset.csv", row.names = FALSE)
print(head(final_data,50))