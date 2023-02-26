library(tidyverse)
library(googlesheets4)
library(class)
library(caret)
library(creditmodel)
rawDF <- read_csv("././KNN-diabetes.csv")
str(rawDF)
summary(rawDF)
sum(duplicated(rawDF))
rawDF$Outcome <- factor(rawDF$Outcome, levels = c(0, 1), labels = c("Negative", "Positive")) %>%  relevel("Positive")
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
cleanDF <- rawDF 
cleanDF[,1:8] <- normalize(cleanDF[,1:(ncol(rawDF)-1)])
summary(cleanDF)
test <- train_test_split(cleanDF)[[1]]
train <- train_test_split(cleanDF)[[2]]
test_feat <- test[-9]
train_feat <- train[-9]
test_labels <- test[, 9]
train_labels <- train[,9]
cleanDF_test_pred <- knn(train = as.matrix(train_feat), test = as.matrix(test_feat), cl = as.matrix(train_labels), k = 21)
head(cleanDF_test_pred)
confusionMatrix(cleanDF_test_pred, test_labels[[1]], positive = NULL, dnn = c("Prediction", "True"))
