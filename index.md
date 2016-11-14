# Practical Machine Learning
Fenton Taylor  
November 9, 2016  



##__I. Introduction__
The purpose of this exercise is to find a prediction model, using machine learning algorithms,
to perform qualitative activity recognition. The data comes from Velloso, et al.'s publication entitled "Qualitative Activity Recognition of Weight Lifting Exercises" as part of their Human Activity Recognition research. Their publications and datasets are free and open to the public and can be found at the following link: [Read more here.](http://groupware.les.inf.puc-rio.br/har#ixzz4PXRAjZK1)

In this particular study, as described on the author's website, "six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E)." Participants were fitted with four accelerometer devices to measure kinetic movement. The devices were located on a waist belt, an upper arm band, a glove, and a dumbbell.

##__II. Getting and Cleaning the Data__
###__A. Load Libraries__


```r
library(caret)
library(rattle)
library(corrplot)
library(knitr)
library(grid)
library(gridExtra)
```

###__B. Download Files__

```r
if(!file.exists("pml-training.csv")){
      download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                    destfile = "pml-training.csv")
}

if(!file.exists("pml-testing.csv")){
      download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                    destfile = "pml-testing.csv")
}

training <- read.csv("pml-training.csv", na.strings = c("", NA))
testing <- read.csv("pml-testing.csv", na.strings = c("", NA))
```

###__C. Transforming the Data__
The data needs to be cleaned so that the variables are all relevant predictors and the machine learning algorithms can process them more efficiently. Both the original training and testing datasets receive the following transformations:

1. Randomly sort the training data so that k-fold cross-validation can be used.
2. Remove any columns with more that 75% NA values.
3. Remove the first 7 columns that contain meta-data about the testing subject and measurements. These variables are not relevant to the prediction.

Finally, using random subsampling, the original training set needs to be split into smaller training and testing sets (60% and 40%, respectively) to build and test the accuracy of the models.


```r
# Transformation 1
myTrain <- training[sample(1:nrow(training)),]

# Transformation 2
myTrain <- myTrain[, colSums(is.na(training)) < nrow(training) * 0.75]
myTest <- testing[, colSums(is.na(testing)) < nrow(testing) * 0.75]

# Transformation 3
myTrain <- myTrain[, -c(1:7)]
myTest <- myTest[, -c(1:7)]

set.seed(432)
inTrain <- createDataPartition(y=myTrain$classe, p=0.6, list=FALSE)
tra <- myTrain[inTrain,]
tes <- myTrain[-inTrain,]
```

##__III. Exploratory Analysis__
The data needs to be checked to see if any further transformation is necessary before attempting to build models with it.

###__A. Basic__
Dimensions of the training and testing set made from the training set:

```r
dim(tra); dim(tes)
```

```
## [1] 11776    53
```

```
## [1] 7846   53
```

Check if there are any NA values left in the traing or test set that need to be imputed. TRUE = no NA.

```r
all(colSums(is.na(tra))==0); all(colSums(is.na(tes))==0)
```

```
## [1] TRUE
```

```
## [1] TRUE
```

Check to see if any variables have near zero variance. If FALSE, then there are no variables with near zero variance.

```r
any(nearZeroVar(tra, saveMetrics = TRUE)$nzv)
```

```
## [1] FALSE
```

###__B. Correlation Analysis__


```r
corrs <- cor(tra[,-53])
corrplot(corrs, method="color", order="original", type="lower", tl.cex = 0.8, tl.col="grey20")
```

<img src="index_files/figure-html/correlation-1.png" style="display: block; margin: auto;" />

There appear to be several variables that are highly correlated (the dark squares in the plot), but the majority of the correlations do not appear to be a cause for concern. Therefore, the variables will remain unaltered for the initial model building.

The remaining variables should now be only measurements collected from the devices (with no NA values) and the class of the exercise performed. The data is now ready to build and test models on.

##__IV. Model Building__
All models will use k-fold cross-validation with $k=5$. 

###__Method 1: Classification Tree__

```r
set.seed(111)
system.time(mod.ct <- train(x=tra[,-53], y=tra$classe, method="rpart",
                trControl=trainControl(method="cv", number=5)))
```

```
##    user  system elapsed 
##    6.34    0.13    6.52
```

```r
saveRDS(mod.ct, file="modelCT.rds")
```


```r
mod.ct$finalModel
```

```
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 10793 7452 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -33.15 964   12 A (0.99 0.012 0 0 0) *
##      5) pitch_forearm>=-33.15 9829 7440 A (0.24 0.23 0.21 0.2 0.12)  
##       10) yaw_belt>=169.5 502   55 A (0.89 0.052 0 0.052 0.006) *
##       11) yaw_belt< 169.5 9327 7086 B (0.21 0.24 0.22 0.2 0.13)  
##         22) magnet_dumbbell_y< 436.5 7778 5789 C (0.24 0.19 0.26 0.2 0.11)  
##           44) magnet_dumbbell_z< -30.5 2338 1258 A (0.46 0.24 0.11 0.13 0.058) *
##           45) magnet_dumbbell_z>=-30.5 5440 3713 C (0.15 0.17 0.32 0.23 0.14) *
##         23) magnet_dumbbell_y>=436.5 1549  768 B (0.038 0.5 0.042 0.23 0.19) *
##    3) roll_belt>=130.5 983    7 E (0.0071 0 0 0 0.99) *
```

```r
# Use model to predict classe in testing set
pred.ct <- predict(mod.ct, newdata=tes)

fancyRpartPlot(mod.ct$finalModel, main = "Classification Tree Using All Variables")
```

![](index_files/figure-html/ct2-1.png)<!-- -->

```r
cm.ct <- confusionMatrix(pred.ct, tes$classe)

# Table of predicted vs. actual exercise classes
cm.ct$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1667  360  171  223   91
##          B   40  527   42  232  200
##          C  520  631 1155  831  515
##          D    0    0    0    0    0
##          E    5    0    0    0  636
```

```r
# Overall accuracy statistics
round(cm.ct$overall, 3)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##          0.508          0.375          0.497          0.519          0.284 
## AccuracyPValue  McnemarPValue 
##          0.000            NaN
```

####__Analysis__
The simple classification tree with all the measurement variables as predictors obtains only 50.8%  out of sample accuracy (out of sample error rate = 49.2%) when applied to the testing set. That is quite poor predictive power, so other algorithms need to be explored.

###__Method 2: Random Forests__


```r
set.seed(222)
system.time(mod.rf <- train(x=tra[,-53], y=tra$classe, method="rf",
                trControl=trainControl(method="cv", number=5)))
```

```
##    user  system elapsed 
##  590.66    3.55  595.47
```

```r
saveRDS(mod.rf, file="modelRF.rds")
```


```r
mod.rf
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9421, 9421, 9421, 9420 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9887906  0.9858187
##   27    0.9886205  0.9856042
##   52    0.9836101  0.9792652
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
# Use model to predict classe in testing set
pred.rf <- predict(mod.rf, newdata=tes)

cm.rf <- confusionMatrix(pred.rf, tes$classe)

# Table of predicted vs. actual exercise classes
cm.rf$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    5    0    0    0
##          B    0 1513    7    0    0
##          C    0    0 1360   10    0
##          D    0    0    1 1275    2
##          E    0    0    0    1 1440
```

```r
# Overall accuracy statistics
round(cm.rf$overall, 3)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##          0.997          0.996          0.995          0.998          0.284 
## AccuracyPValue  McnemarPValue 
##          0.000            NaN
```

####__Analysis__
The random forest algorithm provides a very accurate model with 99.7% out of sample accuracy (out of sample error rate = 0.3%). The only downside is that it is computationally inefficient and not easily interpretable.

###__Method 3: Generalized Boosted Model__

```r
set.seed(333)
system.time(mod.gbm <- train(x=tra[,-53], y=tra$classe, method="gbm",
                trControl=trainControl(method="cv", number=5),
                verbose=FALSE))
```

```
##    user  system elapsed 
##  259.18    0.28  259.90
```

```r
saveRDS(mod.gbm, file="modelGBM.rds")
```


```r
mod.gbm
```

```
## Stochastic Gradient Boosting 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9420, 9422, 9421, 9421, 9420 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7506803  0.6837357
##   1                  100      0.8165767  0.7677470
##   1                  150      0.8532623  0.8142470
##   2                   50      0.8552147  0.8165511
##   2                  100      0.9088830  0.8846818
##   2                  150      0.9337635  0.9161952
##   3                   50      0.8961439  0.8685029
##   3                  100      0.9425101  0.9272611
##   3                  150      0.9614466  0.9512300
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
# Use model to predict classe in testing set
pred.gbm <- predict(mod.gbm, newdata=tes[,-53], n.trees=150)
cm.gbm <- confusionMatrix(pred.gbm, tes$classe)

# Table of predicted vs. actual exercise classes
cm.gbm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2189   31    0    0    2
##          B   30 1454   34    0   19
##          C    7   32 1319   33   19
##          D    2    1   13 1249   14
##          E    4    0    2    4 1388
```

```r
# Overall accuracy statistics
round(cm.gbm$overall, 3)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##          0.969          0.960          0.964          0.972          0.284 
## AccuracyPValue  McnemarPValue 
##          0.000          0.000
```

####__Analysis__
Generalized boosting model gave a fairly good predictive model at 96.9% out of sample accuracy (out of sample error rate = 3.1%). It was not quite as good as random forest, but it was much faster computationally.

###__Variable Importance__
One way to compare the models is to look at the relative importance of the variables.


```r
imp.ct <- varImp(mod.ct)[[1]]
names.ct <- row.names(imp.ct)
imp.ct <- data.frame(varName=names.ct,
                     Overall=imp.ct, 
                     Model = gl(1,k=length(imp.ct),labels="CT"),
                     row.names = NULL)
imp.ct <- imp.ct[order(imp.ct$Overall, decreasing = T),]
imp.ct <- imp.ct[1:15,]
imp.ct$varName <- factor(imp.ct$varName, levels = imp.ct$varName[order(imp.ct$Overall)])

imp.rf <- varImp(mod.rf)[[1]]
names.rf <- row.names(imp.rf)
imp.rf <- data.frame(varName=names.rf, 
                     Overall=imp.rf, 
                     Model = gl(1,k=length(imp.rf),labels="RF"),
                     row.names = NULL)
imp.rf <- imp.rf[order(imp.rf$Overall, decreasing = T),]
imp.rf <- imp.rf[1:15,]
imp.rf$varName <- factor(imp.rf$varName, levels = imp.rf$varName[order(imp.rf$Overall)])

imp.gbm <- varImp(mod.gbm)[[1]]
names.gbm <- row.names(imp.gbm)
imp.gbm <- data.frame(varName=names.gbm, 
                     Overall=imp.gbm, 
                     Model = gl(1,k=length(imp.gbm),labels="GBM"),
                     row.names = NULL)
imp.gbm <- imp.gbm[order(imp.gbm$Overall, decreasing = T),]
imp.gbm <- imp.gbm[1:15,]
imp.gbm$varName <- factor(imp.gbm$varName, levels = imp.gbm$varName[order(imp.gbm$Overall)])

g <- ggplot(imp.gbm, aes(x=varName, y=Overall)) +
      geom_bar(stat="identity", fill=alpha("blue", 0.7)) + 
      labs(list(y="Overall Importance",x="Variable Name",title= "GBM Variable Importance")) +
      theme_classic()+
      coord_flip()
h <- ggplot(imp.rf, aes(x=varName, y=Overall)) +
      geom_bar(stat="identity", fill=alpha("red", 0.7)) + 
      labs(list(y="Overall Importance",x="Variable Name",title= "RF Variable Importance")) +
      theme_classic()+
      coord_flip()
j <- ggplot(imp.ct, aes(x=varName, y=Overall)) +
      geom_bar(stat="identity", fill=alpha("green", 0.5)) + 
      labs(list(y="Overall Importance",x="Variable Name",title= "CT Variable Importance")) +
      theme_classic()+
      coord_flip()
grid.arrange(h,g,j, ncol=2)
```

<img src="index_files/figure-html/varImp-1.png" style="display: block; margin: auto;" />

####__Analysis__
The RF and GBM models use mostly the same variables as important predictors. The first four variables, roll_belt, yaw_belt, magnet_dumbbell_z, magnet_dumbbell_y, are identical for both models. They share 8 of the top 10 and 11 of the top 15 most important variables. The CT model found "roll_belt" to be only the third most important variable, whereas RF and GBM found it to be most important, with GBM heavily emphasizing its importance in relation to the other variables.


##__V. Model Selection and Testing Prediction__

The three models had the following accuracy when used to predict on the testing set that was a subsample of the training set:
      
      1. Classification Tree: 50.8%
      2. Random Forest: 99.7%
      3. Boosting: 96.9%

Random Forest will be used because it has the best accuracy, although boosting did a pretty good job as well. The following are the final predictions associated with their problem id numbers.


```r
pred.final <- predict(mod.rf, myTest)
pred.final.ct <- predict(mod.ct, myTest)
pred.final.gbm <- predict(mod.gbm, myTest)
# After submitting the predictions in the quiz, random forest had 100% accuracy
answers <- data.frame(ProblemID=myTest$problem_id, 
                      Actual=pred.final,
                      rfPrediction=pred.final,
                      rfCheck= pred.final==pred.final,
                      gbmPrediction=pred.final.gbm,
                      gbmCheck=pred.final.gbm==pred.final,
                      ctPrediction=pred.final.ct,
                      ctCheck=pred.final.ct==pred.final)
kable(answers, align=rep('c', ncol(answers)))
```



 ProblemID    Actual    rfPrediction    rfCheck    gbmPrediction    gbmCheck    ctPrediction    ctCheck 
-----------  --------  --------------  ---------  ---------------  ----------  --------------  ---------
     1          B            B           TRUE            B            TRUE           A           FALSE  
     2          A            A           TRUE            A            TRUE           A           TRUE   
     3          B            B           TRUE            B            TRUE           C           FALSE  
     4          A            A           TRUE            A            TRUE           C           FALSE  
     5          A            A           TRUE            A            TRUE           C           FALSE  
     6          E            E           TRUE            E            TRUE           C           FALSE  
     7          D            D           TRUE            D            TRUE           C           FALSE  
     8          B            B           TRUE            B            TRUE           C           FALSE  
     9          A            A           TRUE            A            TRUE           A           TRUE   
    10          A            A           TRUE            A            TRUE           A           TRUE   
    11          B            B           TRUE            B            TRUE           C           FALSE  
    12          C            C           TRUE            C            TRUE           C           TRUE   
    13          B            B           TRUE            B            TRUE           C           FALSE  
    14          A            A           TRUE            A            TRUE           A           TRUE   
    15          E            E           TRUE            E            TRUE           C           FALSE  
    16          E            E           TRUE            E            TRUE           C           FALSE  
    17          A            A           TRUE            A            TRUE           C           FALSE  
    18          B            B           TRUE            B            TRUE           A           FALSE  
    19          B            B           TRUE            B            TRUE           A           FALSE  
    20          B            B           TRUE            B            TRUE           C           FALSE  




##Sources
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.





~~~
