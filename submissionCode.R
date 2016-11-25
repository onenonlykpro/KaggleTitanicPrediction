# KIT: What sorts of people were likely to survive?
# Apply the tools of machine learning to predict which passengers survived the tragedy
require("rpart")
require("caret")
require("randomForest")
require("plyr")
require("inTrees")
require("e1071")
require("forestFloor")
require("RColorBrewer")
require("party")
require("bartMachine")
require("klaR")
require("rpart.plot")

## Load Titanic data to variable and remove NAs
setwd("C:/Users/Kyle/SkyDrive/Kaggle/Titanic - Machine Learning from Disaster/datasets")
titanicData <- read.csv("train.csv")
View(titanicData)
finalTitanicData <- titanicData[complete.cases(titanicData), ]
finalTitanicData$Survived <- as.factor(finalTitanicData$Survived)

## Create variable to use in future models, defining survived as a function of other variables
fitFormula <- as.formula(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked)

## Create test data table
titanicTest <- read.csv("test.csv")
titanicTest$Survived <- NA



### Create and review Conditional Decision Tree
fitCDT <- ctree(fitFormula, 
                data = finalTitanicData)
plot (fitCDT)
summary(fitCDT)



### Create and review Regression Tree model
fitRT <- rpart(fitFormula, 
                 finalTitanicData,
                 method = "anova")
summary(fitRT)
post(fitRT, 
     title. = "Regression Tree Results",
     filename = "",
     digits = 2,
     pretty = TRUE,
     use.n = TRUE)

## Reduce overfitting
RTprune <- fitRT$cptable[which.min(fitRT$cptable[,"xerror"]),"CP"]
newFitRT <- prune(fitRT, cp=RTprune)
summary(newFitRT)
post(newFitRT, 
     title. = "Pruned Regression Tree Results",
     filename = "",
     digits = 2,
     pretty = TRUE,
     use.n = TRUE)



### Create and review Classification Tree model
fitCT <- rpart(fitFormula, 
               finalTitanicData,
               method = "class")
summary(fitCT)
post(fitCT, 
     title. = "Regression Tree Results",
     filename = "",
     digits = 2,
     pretty = TRUE,
     use.n = TRUE)

## Reduce overfitting
CTprune <- fitCT$cptable[which.min(fitCT$cptable[,"xerror"]),"CP"]
newFitCT <- prune(fitCT, cp=CTprune)
summary(newFitCT)
post(newFitCT, 
     title. = "Pruned Classification Tree Results",
     filename = "",
     digits = 2,
     pretty = TRUE,
     use.n = TRUE)



### Create and review Naive Bayes
fitNB <- train(fitFormula, 
                 finalTitanicData,
                 method = "nb")
fitNB
summary(fitNB)
plot(fitNB)


## Test the models

?predict
