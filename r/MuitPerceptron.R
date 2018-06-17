
#      w1
# X1 > w2 b1 > w5
#                 b3 > Sigmoid() > Y
# X2 > w3 b2 > w6
#      w4
learning_rate <- 0.3

w1=0
w2=0
w3=0
w4=0
w5=0
w6=0
b1=0
b2=0
b3=0

#Sigmoid
Sigmoid <- function(x){
  1/(1+exp(-x))
}
#The Neural 2*2*1
NeuralTrain<- function(X1,X2,answer){
  #input layer -> hidden layer
  #n1 = Sigmoid( w1*X1 + w2*X2 + b1 );
  n1 <- X1*w1+X2*w2+b1
  h1 <- Sigmoid(n1)
  #n2 = Sigmoid( w3*X1 + w4*X2 + b2 );
  n2 <- X1*w3+X2*w4+b2
  h2 <- Sigmoid(n2)
  #hidden layer -> output layer
  #output = Sigmoid( w5*n1 + w6*n2 + b3 );
  n3 <- h1*w5+h2*w6+b3
  output <- Sigmoid(n3)
  #backward path
  #output layer->hidden layer
  #Use eval.parent(substitute(Values)) update variable to Global Environment
  #learning_rate*(answer-output)*output*(1-output)*h1
  eval.parent(substitute(w5 <- w5+ learning_rate * (answer - output)*output*(1-output)*h1)) 
  eval.parent(substitute(w6 <- w6+learning_rate * (answer - output)*output*(1-output)*h2))
  eval.parent(substitute(b3 <- b3+ learning_rate * (answer - output)*output*(1-output)*1))
  #hidden layer->input layer
  eval.parent(substitute(w1 <- w1 + learning_rate * (answer - output)*output*(1-output)*w5*(h1)*(1-h1)*X1))
  eval.parent(substitute(w2 <- w2 + learning_rate * (answer - output)*output*(1-output)*w5*(h1)*(1-h1)*X2))
  eval.parent(substitute(w3 <- w3 + learning_rate * (answer - output)*output*(1-output)*w6*(h2)*(1-h2)*X1))
  eval.parent(substitute(w4 <- w4 + learning_rate * (answer - output)*output*(1-output)*w6*(h2)*(1-h2)*X2))
  eval.parent(substitute(b2 <- b2 + learning_rate * (answer - output)*output*(1-output)*w6*(h2)*(1-h2)*1))
  eval.parent(substitute(b1 <- b1 + learning_rate * (answer - output)*output*(1-output)*w5*(h1)*(1-h1)*1))
 
  return (answer - output)
}
Neural <- function(X1,X2){
  n1 <- X1*w1+X2*w2+b1
  h1 <- Sigmoid(n1)
  n2 <- X1*w3+X2*w4+b2
  h2 <- Sigmoid(n2)
  #hidden layer -> output layer
  #output = Sigmoid( w5*n1 + w6*n2 + b3 );
  n3 <- h1*w5+h2*w6+b3
  output <- Sigmoid(n3)
  return(output)
}


#init
w1 =runif(1)
w2 =runif(1)
b1 =runif(1)


# learn AND
# this will converge
set.seed(10)
for(i in c(1:300000)){
  x=round(runif(1)*10) %% 2
  y=round(runif(1)*10) %% 2
  z=as.double(x&&y)
  output = NeuralTrain(x,y,z)
  print(output)
}
print("1 AND 1 :")
print(Neural(1,1))
print("1 AND 0 :")
print(Neural(1,0))
print("0 AND 1 :")
print(Neural(0,1))
print("0 AND 0 :")
print(Neural(0,0))

# learn OR
# this will converge
set.seed(15)
for(i in c(1:30000)){
  x=round(runif(1)*10) %% 2
  y=round(runif(1)*10) %% 2
  z=as.double(x|y)
  print(NeuralTrain(x,y,z))
}
print("1 OR 1 :")
print(Neural(1,1))
print("1 OR 0 :")
print(Neural(1,0))
print("0 OR 1 :")
print(Neural(0,1))
print("0 OR 0 :")
print(Neural(0,0))

# learn XOR
# this will converge
set.seed(10)
for(i in c(1:30000)){
  x=round(runif(1)*10) %% 2
  y=round(runif(1)*10) %% 2
  z=ifelse(xor(x,y)==TRUE,1,0)
  print(NeuralTrain(x,y,z))
}

print("1 XOR 1 :")
print(Neural(1,1))
print("1 XOR 0 :")
print(Neural(1,0))
print("0 XOR 1 :")
print(Neural(0,1))
print("0 XOR 0 :")
print(Neural(0,0))


