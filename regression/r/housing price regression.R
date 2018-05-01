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

data = read.csv("/Users/Janet/Desktop/EBAC/courses/???BS???5206 COMPUTATIONAL INTELLIGENCE/assignment/data_cleaned.csv")

##select variables
data = data %>%
  subset(select = c(sale_price,
                    block,
                    lot,
                    residential_units,
                    commercial_units,
                    total_units,
                    land_sqft,
                    gross_sqft,
                    # tax_class_at_sale,
                    LotArea,
                    BldgArea,
                    ComArea,
                    ResArea,
                    OfficeArea,
                    RetailArea,
                    GarageArea,
                    StrgeArea,
                    FactryArea,
                    OtherArea,
                    NumBldgs,
                    NumFloors,
                    UnitsRes,
                    UnitsTotal,
                    AssessLand,
                    YearBuilt,
                    YearAlter1,
                    YearAlter2,
                    BuiltFAR,
                    ResidFAR,
                    CommFAR,
                    FacilFAR)) %>%
  filter(YearBuilt !=0)
# data.Normalization(type = "n3")

## split dataset into traning and test datasets
set.seed(1234)
splitData = sample.split(data$sale_price, SplitRatio = 0.7)
train = data[splitData,]
test = data[!splitData,]


## normalize training dataset
train_x = train %>% subset(select = -c(sale_price))
train_y = train %>% subset(select = c(sale_price))
train_scale = data.Normalization(train_x, type = "n3")

##normalize test dataset using the same parameters and the function as training dataset
test_x = test %>% subset(select = -c(sale_price))
test_y = test %>% subset(select = c(sale_price))

train_mean = colMeans(train_x)
train_range = colRanges(as.matrix(train_x)) %>%
  as.data.frame() %>%
  mutate(range = V2 - V1)

test_x[4832,] = train_mean
test_x[4833,] = t(train_range$range)
test_scale = as.data.frame(apply(test_x, 2, function(x) (x-x[4832])/x[4833]))
test_scale = test_scale[1:4831,]

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

model = keras_model_sequential()
model %>%
  layer_dense(units = ncol(train_scale), input_shape = c(ncol(train_scale)), activation = "sigmoid", kernel_initializer='normal') %>% 
  layer_dense(units = 6, activation = "sigmoid", kernel_initializer='normal') %>% 
  layer_dense(units = 1, activation = "sigmoid", kernel_initializer='normal')

summary(model)

## evaluate model
model %>% compile(
  optimizer = "sgd",
  loss = "mse"
)

result = model %>%
  fit(as.matrix(train_scale), as.matrix(train_y), epochs = 100, batch_size = 1000, validation_split = 0.2)
plot(result)



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


# #######################  MLFF keras  ###########################
# train_target = traindata$sale_price
# traindata = traindata %>% subset(select = -c(sale_price))
# model = keras_model_sequential()
# model %>%
#   layer_dense(units = ncol(traindata), input_shape = c(ncol(traindata)), activation = "sigmoid", kernel_initializer='normal') %>% 
#   layer_dense(units = 6, activation = "sigmoid", kernel_initializer='normal') %>% 
#   layer_dense(units = 1, activation = "sigmoid", kernel_initializer='normal')
# 
# summary(model)
# 
# ## evaluate model
# model %>% compile(
#   optimizer = "sgd",
#   loss = "mse"
# )
# 
# track = model %>%
#   fit(as.matrix(traindata), train_target, epochs = 100, batch_size = 100, validation_split = 0.2)
# ## get the weight in the model
# 
# class(track)
# plot(track)
