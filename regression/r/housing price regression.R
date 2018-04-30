library(dplyr)
library(nnet)
library(NeuralNetTools)
library(caTools) ##split data
library(tidyverse) 
library(kerasR)
library(tensorflow)
library(keras)
library(clusterSim)

rm(list = ls())

# data_raw = read.csv("/Users/Janet/Desktop/EBAC/courses/???BS???5206 COMPUTATIONAL INTELLIGENCE/assignment/brooklynhomes2003to2017/brooklyn_sales_map.csv")

# attach(data_raw)
# attribute_orignal = names(data_raw)
# attribute_selected = subset(data_raw, select = -c(borough, easement, address, apartment_number, zip_code))

data = read.csv("/Users/alexjzy/Desktop/Py-Projects/NUS-CI1-CA1/regression/r/data_cleaned.csv")


data = data %>%
  subset(select=c(sale_price,
                  land_sqft,
                  total_units,
                  gross_sqft,
                  residential_units,
                  commercial_units,
                  YearBuilt)) %>%
  # filter(sale_price > 1000000) %>%
  data.Normalization(type = "n3")


## split dataset into traning and test datasets
set.seed(1234)
splitData = sample.split(data$sale_price, SplitRatio = 0.7)
traindata = data[splitData,]
testdata = data[!splitData,]


######################### PCA #############################
# ##correlation matrix
# cor(traindata)
# 
# ## PCA on training dataset
# pca = princomp(traindata, scores = TRUE, cor = TRUE)
# summary(pca)
# 
# ##scree plot of eigenvalues
# screeplot(pca, type = "line", main = "Scree Plot")


#######################  MLFF keras  ###########################
train_target = traindata$sale_price
traindata = traindata %>% subset(select = -c(sale_price))
model = keras_model_sequential()
model %>%
  layer_dense(units = ncol(traindata), input_shape = c(ncol(traindata)), activation = "sigmoid", kernel_initializer='normal') %>% 
  layer_dense(units = 4, activation = "sigmoid", kernel_initializer='normal') %>% 
  layer_dense(units = 1, activation = "sigmoid", kernel_initializer='normal')

summary(model)

## evaluate model
model %>% compile(
  optimizer = "sgd",
  loss = "mse"
)

track = model %>%
  fit(as.matrix(traindata), train_target, epochs = 100, batch_size = 100)
## get the weight in the model

class(track)
plot(track)
