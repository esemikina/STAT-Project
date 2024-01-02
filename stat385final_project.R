school <- read.csv("/Users/liza/Desktop/stat385 project/Chicago_Public_Schools_-_Progress_Report_Cards__2011-2012_.csv")
library(dplyr)

#########################################################
### Plotting the dependent variable for MLR
#########################################################

# graduation rate: 
grad_rate <- school$Graduation.Rate..[school$Graduation.Rate.. != "NDA"]
grad_rate = as.numeric(grad_rate)

### Descriptive statistics
hist(grad_rate, main = "Distribution of Graduation Rates", xlab = "Graduation Rate")


breaks <- c(0, 50, 60, 70, 80, 100)
labels <- c("0-50", "51-60", "61-70", "71-80", "81-100")
grad_rate_categories <- cut(grad_rate, breaks = breaks, labels = labels)
barplot(table(grad_rate_categories), main = "Barplot of Graduation Rate Categories", xlab = "Graduation Rate Category", ylab = "Frequency")

#########################################################
### Cleaning the data
#########################################################
school[school == "NDA"] <- NA

# splitting into hs
high_schools = school[school$`Elementary..Middle..or.High.School` == "HS", ]

# removing columns that are not needed or have all NAs
high_schools = high_schools[, -c(1:11)]

high_schools <- high_schools %>%
  mutate_all(funs(type.convert(as.character(.))))

columns_to_delete <- c(26:45)
high_schools <- high_schools[, -columns_to_delete]

high_schools = na.omit(high_schools)

## Convert to correct types 
high_schools <- high_schools %>%
  mutate_all(funs(type.convert(as.character(.))))


#########################################################
### Random Forest Feature Importance
#########################################################
library(randomForest)

# Tune 
mtry <- tuneRF(high_schools,high_schools$Graduation.Rate.., ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

set.seed (1)
rand.forest_model <- randomForest(Graduation.Rate.. ~ ., mtry = best.m, data = high_schools, importance = TRUE)
rand.forest_model

# Get variable importance
var_importance <- importance(rand.forest_model)
var_importance
varImpPlot(rand.forest_model, main = "Feature Importance")
# MeanDecreaseAccuracy and Mean Decrease Gini 
# The higher the value of mean decrease accuracy or mean decrease gini score, the higher the importance of the variable to our model.

top_features <- rownames(var_importance)[order(-var_importance[, 1])][1:10]



#########################################################
# Modifying Data based on feature importance
#########################################################

#### High school dataset with selected variables 
high_schools_selected = school[school$`Elementary..Middle..or.High.School` == "HS", ]
high_schools_selected =  high_schools_selected[, c(top_features, "Graduation.Rate..")]
high_schools_selected = na.omit(high_schools_selected)

high_schools = high_schools_selected

str(high_schools)
## Convert to correct types 
high_schools <- high_schools %>%
  mutate_all(funs(type.convert(as.character(.))))

#Delete because only observation containing "Strong", causing errors in split training and test sets
high_schools <- high_schools[, !colnames(high_schools) %in% c("Teachers.Icon")]


#########################################################
### MLR
#########################################################

model <- lm(Graduation.Rate.. ~ ., data = high_schools)
summary(model)

plot(model)


library(car)
vif(model)
# GVIF Df GVIF^(1/(2*Df))
# Instruction.Score               2.529279  1        1.590371
# X11th.Grade.Average.ACT..2011. 55.797227  1        7.469754
# College.Enrollment.Rate..       4.983752  1        2.232432
# X10th.Grade.PLAN..2009.        72.787685  1        8.531570
# X10th.Grade.PLAN..2010.        52.775620  1        7.264683
# Safety.Icon                    69.556535  4        1.699387
# Average.Student.Attendance      2.425086  1        1.557269
# Safety.Score                   22.395530  1        4.732392
# X9th.Grade.EXPLORE..2010.      33.286677  1        5.769461


#########################################################
#Bootstrap
install.packages("boot")
library(boot)

# Function to obtain coefficients from linear model
boot_fn <- function(data, indices) {
  # Sample with replacement from the dataset
  d <- data[indices,]
  
  # Fit the model
  fit <- lm(Graduation.Rate.. ~ ., data = d)
  
  # Return the coefficients
  return(coef(fit))
}

high_schools_boot = high_schools
boot_results <- boot(data = high_schools, statistic = boot_fn, R = 1000)

# Perform bootstrap with R = 1000 resamples
set.seed(123) # for reproducibility
boot_results <- boot(high_schools_boot, boot_fn, R = 1000)

# Display the bootstrap results
print(boot_results)

# Calculate basic bootstrap confidence intervals
boot_ci <- boot.ci(boot_results, type = "basic", index = 1) 
print(boot_ci)
#########################################################
#Cross-Validation for MLR
library(caret)

# Set the seed
set.seed(123)

# Define control parameters for cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Define the formula
formula <- Graduation.Rate.. ~ .

# Train the model using cross-validation
cv_model <- train(formula, data = high_schools, method = "lm", trControl = train_control)

# results of the cross-validation
print(cv_model)

# summary of the best model
print(summary(cv_model$finalModel))


#########################################################
### - Classification -----------------------------------------------------------
#########################################################
#Create Qualitative Variable
# - https://www.cps.edu/globalassets/cps-pages/about-cps/finance/budget/budget-2014/appendixa.pdf#:~:text=Using%20the%20CPS%20measure%2C%20the%202012-2013%20CPS%20graduation,with%20the%20prior%20year%20rate%20of%2058.3%20percent.
# - "Using the CPS measure, the 2012-2013 CPS graduation rate was 61.2 percent, which represents a 2.9 percent increase compared with the prior year rate of 58.3 percent."
high_schools$Tier <- 0
high_schools$Tier[high_schools$Graduation.Rate.. >= 58.3] <- "Above Average"
high_schools$Tier[high_schools$Graduation.Rate.. < 58.3] <- "Below Average"
high_schools$Tier <- factor(high_schools$Tier)
high_schools$Safety.Icon <- factor(high_schools$Safety.Icon)
high_schools <- high_schools[, !colnames(high_schools) %in% c("Graduation.Rate..")]
#Delete because only observation containing "Strong", causing errors in split training and test sets
high_schools <- high_schools[, !colnames(high_schools) %in% c("Teachers.Icon")]



#Creating training and test sets for logistic Regression
library(caret)
set.seed(2)
trainIndex <- createDataPartition(high_schools$Tier, p = .7, list = FALSE, times = 1)
train <- high_schools[trainIndex, ]
test <- high_schools[-trainIndex, ]


###########################
# - Logistic Regression - #
###########################
# Fit a logistic regression model to the training data
log_model <- glm(Tier ~ ., data =  train, family = "binomial")
summary(log_model)

# Make predictions on the test data
log_predictions <- predict(log_model, newdata = test, type = "response")
log_predictions <- ifelse(log_predictions > 0.5, "Above Average", "Below Average")

# Convert variables to factors with the same levels
log_predictions <- as.factor(log_predictions)
test$Tier <- as.factor(test$Tier)

levels(log_predictions)
levels(test$Tier)

confusionMatrix(log_predictions, test$Tier)


###########################
# - K-Nearest Neighbors - #
###########################
library(class)
knn.pred2<- knn(train[,-9], test[,-9], train[,9], k=2)
knn.pred3<- knn(train[,-9], test[,-9], train[,9], k=3)
knn.pred4<- knn(train[,-9], test[,-9], train[,9], k=4)
knn.pred5<- knn(train[,-9], test[,-9], train[,9], k=5)
knn.pred6<- knn(train[,-9], test[,-9], train[,9], k=6)
knn.pred7<- knn(train[,-9], test[,-9], train[,9], k=7)
knn.pred8<- knn(train[,-9], test[,-9], train[,9], k=8)


# Convert variables to factors with the same levels
knn.pred2 <- as.factor(knn.pred2)
knn.pred3 <- as.factor(knn.pred3)
knn.pred4 <- as.factor(knn.pred4)
knn.pred5 <- as.factor(knn.pred5)
knn.pred6 <- as.factor(knn.pred6)
knn.pred7 <- as.factor(knn.pred7)
knn.pred8 <- as.factor(knn.pred8)
test$Tier <- as.factor(test$Tier)


confusionMatrix(knn.pred2, test$Tier)
confusionMatrix(knn.pred3, test$Tier)
confusionMatrix(knn.pred4, test$Tier)
confusionMatrix(knn.pred5, test$Tier)
confusionMatrix(knn.pred6, test$Tier)
confusionMatrix(knn.pred7, test$Tier)
confusionMatrix(knn.pred8, test$Tier)
