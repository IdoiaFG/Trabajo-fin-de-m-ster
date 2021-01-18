# Paquetes
library(keras)
library(EBImage)
library(tensorflow)

# Se vuelcan las imágenes de la carpeta train a una lista
lista_CNN_train<-list.files("C:/Users/idofu/Desktop/TFM/db_CNN/train", 
                            all.files = FALSE, full.names = TRUE)

# Se crea la lista train cogiendo de cada animal las primeras 210 imágenes
train<-list()
for (i in 1:840){
  train[[i]]<-readImage(lista_CNN_train[i])}
lista_CNN_train<-list()

# Se vuelcan las imágenes de la carpeta test a una lista
lista_CNN_test<-list.files("C:/Users/idofu/Desktop/TFM/db_CNN/test", 
                           all.files = FALSE, full.names = TRUE)

#  Se crea la lista test con 90 imágenes de cada especie
test<-list()
for (i in 1:360){
  test[[i]]<- (readImage(lista_CNN_test[i]))}
lista_CNN_test<-list()

# Se crean las etiquetas de las imágenes y se codifican con One Hot Encoding
trainy <- rep(0:3,c(210,210,210,210))
testyact <- rep(0:3,c(90,90,90,90))
trainy <- to_categorical(trainy)
testy <- to_categorical(testy)

# Preprocesado de los datos de train y test
x <- array(rep(0, 840*224*224*3), dim = c(840, 224, 224, 3))
for (i in 1:840) {x[i,,,] <- resize(train[i,,,], 224, 224)}
trainx <- imagenet_preprocess_input(x)

x <- array(rep(0, 360*224*224*3), dim = c(360, 224, 224, 3))
for (i in 1:360) {x[i] <- resize(test[i], 224, 224)}
testx <- imagenet_preprocess_input(x)

# Modelo con RESNET50
pretrained <- application_resnet50(weights = 'imagenet',
                                   include_top = FALSE,
                                   input_shape = c(224, 224, 3))

model <- keras_model_sequential() %>% 
         pretrained %>% 
         layer_flatten() %>% 
         layer_dense(units = 256, activation = 'relu') %>% 
         layer_dense(units = 4, activation = 'softmax')
freeze_weights(pretrained)

# Función compile
model %>% compile(loss = "categorical_crossentropy",
                  optimizer = 'adam',
                  metrics = 'accuracy')

# Entrenamiento del modelo
history <- model %>% fit(trainx,
                         trainy,
                         epochs = 45,
                         batch_size = 10,
                         validation_split = 0.2)

# Evaluación y predicción
model %>% evaluate(testx, testy)
pred <- model %>% predict_classes(testx)
table(Predicted = pred, Actual = testyact )
