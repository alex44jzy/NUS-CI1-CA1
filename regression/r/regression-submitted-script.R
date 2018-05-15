library(dplyr)
library(nnet)
library(NeuralNetTools)
library(caTools) ##split data
library(tidyverse) 
library(kerasR)
library(tensorflow)
library(keras)
library(clusterSim)
library(caret)
library(recipes)
library(matrixStats)
library(corrplot)
library(ade4)
library(reshape2)
library(data.table)
library(nnet)
library(NeuralNetTools)
library(grnn)
library(RSNNS)
library(ggplot2)
library(neuralnet)
library(MLmetrics)

rm(list = ls())

# maybe modify by yourself
data = read.csv("/Users/Janet/Desktop/111.csv")

################### Data Exploration ########################

summary(data)

## narrow sales price range
data = data %>% 
  filter(sale_price>100000 & sale_price < 3000000)



# exploration and log-transform of y
ggplot(data=data[!is.na(data$sale_price),], aes(x=sale_price)) +
  geom_histogram(fill="blue", bins = 50)

data$sale_price = log(data$sale_price)

ggplot(data=data[!is.na(data$sale_price),], aes(x=sale_price)) +
  geom_histogram(fill="blue", bins = 50)



## select attributes
data = subset(data, select = -c(SanitBoro, 
                                Easements, 
                                ComArea, 
                                StrgeArea, 
                                GarageArea, 
                                OfficeArea,
                                RetailArea,
                                AreaSource,
                                OtherArea,
                                FactryArea,
                                ExemptLand,
                                ExemptTot,
                                YearAlter1,
                                YearAlter2,
                                CommFAR,
                                IrrLotCode_nan,
                                SplitZone_nan,
                                tax_class_nan
))



## select attributes using correlation coefficient 
data_numeric = cbind(data[,1:44], data$sale_price)
cor = as.data.frame(cor(data_numeric))
cor_y = as.matrix(cor[,'data$sale_price'])
cor_y = as.data.frame(cor_y, row.names = rownames(cor))
cor_idx = names(which(apply(cor_y, 
                            1,
                            function(x) (x > 0.25 | x < -0.25))))

corrplot(as.matrix(cor[cor_idx, cor_idx]),
         type = 'upper',
         method = 'color',
         addCoef.col = 'black',
         tl.cex = 0.6,
         cl.cex = 0.6,
         number.cex = 0.6)

data_temp =  data[,45:56]
data = cbind(data[,which(colnames(data) %in% cor_idx)], data_temp)



## outlier detection using boxplot and remove outliers
p = ggplot(stack(data[,1:44]), aes(x = ind, y = values)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90)) +
  geom_hline(yintercept = 4, color = 'red') +
  geom_hline(yintercept = -4, color = 'red')
p

data_x[abs(data_x) > 4] = NA
data = cbind(data_x, data_y)
data = na.omit(data)
summary(data)


## split training, validation and test dataset
set.seed(1234)
splitData = sample.split(data$sale_price, SplitRatio = 0.7)
train = data[splitData,]
test = data[!splitData,]


train_x = train %>% subset(select = -c(sale_price))
train_y = train %>% subset(select = c(sale_price))

test_x = test %>% subset(select = -c(sale_price))
test_y = test %>% subset(select = c(sale_price))


set.seed(1234)
ind = sample(seq_len(nrow(train)), size = nrow(train)*0.7)
validation = train[-ind,]
train = train[ind,]

train_x = train %>% subset(select = -c(SalePrice))
train_y = train %>% subset(select = c(SalePrice))

validation_x = validation %>% subset(select = -c(SalePrice))
validation_y = validation %>% subset(select = c(SalePrice))


##################### Model Building MLFF-BP #######################


## PCA
PCA on training dataset
pca = princomp(data[,1:25], scores = TRUE, cor = TRUE, scale = FALSE, center = TRUE)
summary(pca)

##scree plot of eigenvalues
screeplot(pca, type = "line", main = "Scree Plot")

data_pca = cbind(pca$scores[,1:9], data[,26:37])


## MLFF-BP model building using Keras
model = keras_model_sequential()
model %>%
  layer_dense(units = ncol(train_x), input_shape = c(ncol(train_x)), activation = "relu", kernel_initializer='normal') %>% 
  layer_dense(units = 3, activation = "relu", kernel_initializer='normal') %>% 
  # layer_dense(units = 2, activation = "relu", kernel_initializer='normal') %>%
  layer_dense(units = 1, activation = "relu", kernel_initializer='normal')

summary(model)

## evaluate model
model %>% compile(
  optimizer = "adam",
  loss = "mse"
)

result = model %>%
  fit(as.matrix(train_x), as.matrix(train_y),
      epochs = 200, 
      batch_size = 100, 
      validation_data = list(as.matrix(validation_x), as.matrix(validation_y))
      )
plot(result)

## MSE using test dataset
score = model %>% evaluate(as.matrix(test_x), as.matrix(test_y))
score

## predictive value using test dataset
pred = predict(model,as.matrix(test_x))
qplot(as.matrix(exp(pred)), as.matrix(exp(test_y))) +
  xlim(90000, 3000000) +
  ylim(90000, 3000000) +
  geom_point(color='blue') +
  geom_smooth(method = "lm", se = FALSE, color = 'red') +
  xlab('Predictive Price') +
  ylab('Actual Price')

# 
# a = test[1:500,]
# n <- names(a)
# f <- as.formula(paste("sale_price ~", paste(n[!n %in% "sale_price"], collapse = " + ")))
# model = neuralnet(f, a, hidden = c(3), linear.output = T)
# 
# plot(model)


##################### Model Building RBF #######################

## model building
rbf = rbf(as.matrix(train_x), as.matrix(train_y),
          size = 36,
          maxit = 1000,
          initFuncParams = c(0, 1, 0, 0.01, 0.01),
          learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8), 
          linOut=TRUE)

## model evaluation
pred_rbf_test = predict(rbf, test_x)
rbf_mae_test = MSE(pred_rbf_test, as.matrix(test_y))
rbf_mae_test

# b = cbind(exp(pred_rbf_test), exp(test_y))
# b = b[-which(b$sale_price < 1000000 & b$`exp(pred_rbf_test)` > 2000000),]
# b = b[-which(b$sale_price > 2500000 & b$`exp(pred_rbf_test)` < 2000000),]
ggplot(b,aes(`exp(pred_rbf_test)`,sale_price)) +
  xlim(90000, 3000000) +
  ylim(90000, 3000000) +
  geom_point(color='blue') +
  geom_smooth(method = "lm", se = FALSE, color = 'red') +
  xlab('Predictive Price') +
  ylab('Actual Price')


##################### Model Ensemble #######################

pre_avg_test = 0.5*pred + 0.5*pred_rbf_test

ensemble_mse_test = MSE(pre_avg_test, as.matrix(test_y))
ensemble_mse_test
# ensemble_mae_val = MAE(pre_avg_val, as.matrix(validation_y))
# ensemble_mae_val
# ensemble_mae_test = MSE(pre_avg_test, as.matrix(test_y))
# ensemble_mae_test

# 
a = cbind(exp(pre_avg_train),exp(test_y))
a = subset(a, select = -which(sale_price < 500000 & exp(pre_avg_train) >1000000))
ggplot(a, aes(`exp(pre_avg_train)`,sale_price)) +
  xlim(90000, 2000000) +
  ylim(90000, 2000000) +
  geom_point(color='blue') +
  geom_smooth(method = "lm", se = FALSE, color = 'red') +
  xlab('Predictive Price') +
  ylab('Actual Price')

