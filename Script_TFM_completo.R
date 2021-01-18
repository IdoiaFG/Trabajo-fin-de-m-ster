
library(keras)
library(EBImage)
library(tensorflow)

# Función para segmentar las imágenes y obtener las máscaras
quitar_fondo<-function (im){
  im<-resize(im, 264, 264)
  gris<-channel(im, "grey")
  ot<-otsu(gris, range = c(0,1), levels = 256)
  if (ot>=0.5){tresh<-gris>ot } else  {tresh<-gris<ot}
  kernel<- makeBrush(1.5, shape = "disc")
  open<- opening(tresh, kern = kernel) 
  im_fin<-im-toRGB(open) 
return(im_fin)
}

# Cargo las imágenes en una lista  
lista<-list.files("C:/Users/idofu/Desktop/TFM/db_TFM", 
                    all.files = FALSE, full.names = TRUE)
  
# Voy guardando las imágenes sin fondo en la carpeta db_CNN
for (i in 1:length(lista)){
  imagen<-readImage(lista[i])
  filename<-paste0("C:/Users/idofu/Desktop/TFM/db_CNN/imagen",i,".jpg")
  writeImage(quitar_fondo(imagen), filename)
  }
  
# Vuelco las imágenes de la carpeta train a una lista
lista_CNN_train<-list.files("C:/Users/idofu/Desktop/TFM/db_CNN/train", 
                        all.files = FALSE, full.names = TRUE)

# Creo el train cogiendo de cada animal las primeras 210 imágenes
train<-list()
for (i in 1:840){train[[i]]<-readImage(lista_CNN_train[i])}
lista_CNN_train<-list()

# Vuelco las imágenes de la carpeta test a una lista
lista_CNN_test<-list.files("C:/Users/idofu/Desktop/TFM/db_CNN/test", 
                        all.files = FALSE, full.names = TRUE)
  
# Creo el test con 90 imágenes de cada especie
test<-list()
for (i in 1:360){test[[i]]<- (readImage(lista_CNN_test[i]))}
lista_CNN_test<-list()

# Combino las imágenes
train<-abind(train, along=4)
test<-abind(test, along=4)
  
# Redimensiono la combinación
train <-aperm(train, c(4, 1, 2, 3))
test <-aperm(test, c(4, 1, 2, 3))

# Etiqueto las imágenes
trainy <- rep(0:3,c(210,210,210,210))
testy <- rep(0:3,c(90,90,90,90))
  
# One hot encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)
 
# Creo el modelo
model <- keras_model_sequential()
  
model %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3),
                activation = 'relu',
                input_shape = c(264,264,3)) %>%
  layer_conv_2d(filters = 32,
        kernel_size = c(3,3),
        activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64,
                  kernel_size = c(3,3),
                  activation = 'relu') %>%
  layer_conv_2d(filters = 64,
                  kernel_size = c(3,3),
                  activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate=0.25) %>%
  layer_dense(units = 4, activation = 'softmax') %>%
    
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_sgd(lr = 0.01,
                                    decay = 1e-6,
                                    momentum = 0.9,
                                    nesterov = T),
          metrics = c('accuracy'))

summary(model)
  
# Entrenamiento del modelo
model %>%
    fit(train,
        trainLabels,
        epochs = 45,
        batch_size = 32,
        validation_split = 0.2,
        validation_data = list(test, testLabels))

save_model_tf(model, "cnn-TFM")
model <- load_model_tf("C:/Users/idofu/Desktop/TFM/cnn-TFM/")

# Evaluación y predicción con los datos de entrenamiento
model %>% evaluate(train, trainLabels)
pred <- model %>% predict_classes(train)
table(Predicted = pred, Actual = trainy)
  

# Evaluación y predicción con los datos de test
model %>%evaluate(test, testLabels) 
pred <- model %>% predict_classes(test)
table(Predicted = pred, Actual = testy)




