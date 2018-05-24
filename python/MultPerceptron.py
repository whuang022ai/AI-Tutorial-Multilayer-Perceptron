# -*- coding: utf-8 -*-
"""
Created on Fri May 25 02:45:40 2018

@author: lovedoglion

x1   w1    n1->h1
     w2               w5
                           n3     Y 
     w3               w6
x2   w4    n2->h2
"""
import math,random
import matplotlib.pyplot as plt
from datetime import datetime

#學習速率
eta=0.7

w1=0
w2=0
w3=0
w4=0
w5=0
w6=0

b1=0
b2=0
b3=0

#繪圖參數
pltX=[]#x軸為迭代次數
pltY=[]#y軸為誤差值

#繪圖函數
def plotData(plt, data):
    x = [p[0] for p in data]
    y = [p[1] for p in data]
    plt.plot(x, y)

#Neural 2*2*1 input;neural;output
def neural(x1, x2, d):
    global w1,w2,w3,w4,w5,w6,b1,b2,b3
    #forward
    #input layer -> hidden layer
    n1=w1*x1+w2*x2+b1
    n2=w3*x1+w4*x2+b2
    h1=1/(1+math.exp(-1*n1))
    h2=1/(1+math.exp(-1*n2))
    #hidden layer -> output
    n3=w5*h1+w6*h2+b3
    Y=1/(1+math.exp(-1*n3))
    
    #backward path
    #output layer -> hidden layer
    w5+=eta*(d-Y)*Y*(1-Y)*h1
    w6+=eta*(d-Y)*Y*(1-Y)*h2
    b3+=eta*(d-Y)*Y*(1-Y)*1
    #hidden layer -> input layer
    w1+=eta*(d-Y)*Y*(1-Y)*w5*(h1)*(1-h1)*x1
    w2+=eta*(d-Y)*Y*(1-Y)*w5*(h1)*(1-h1)*x2
    w3+=eta*(d-Y)*Y*(1-Y)*w6*(h2)*(1-h2)*x1
    w4+=eta*(d-Y)*Y*(1-Y)*w6*(h2)*(1-h2)*x2
    b1+=eta*(d-Y)*Y*(1-Y)*w5*(h1)*(1-h1)*1
    b2+=eta*(d-Y)*Y*(1-Y)*w6*(h2)*(1-h2)*1
    pltY.append(d-Y)
    return d-Y

#learn AND
def neuralAND():
    x=random.randint(0,1)
    y=random.randint(0,1)
    z=x&y
    return print('neuralAND：',neural(x,y,z))

# OR
def neuralOR():
    x=random.randint(0,1)
    y=random.randint(0,1)
    z=x|y
    return print('neuralOR：',neural(x,y,z))

#learn XOR
def neuralXOR():
    x=random.randint(0,1)
    y=random.randint(0,1)
    z=x^y #XOR位元運算
    return print('neuralXOR：',neural(x,y,z))

def init():
    global w1,w2,w3,w4,w5,w6,b1,b2,b3
    w1=random.random()
    w2=random.random()
    w3=random.random()
    w4=random.random()
    w5=random.random()
    w6=random.random()
    b1=random.random()
    b2=random.random()
    b3=random.random()
    
#main()    
random.seed()
init()
for i in range(8000):  
    neuralXOR()
    pltX.append(i)
    
"""
繪圖
"""
#將X軸
Chart=list(zip(pltX,pltY))
plotData(plt, Chart)#X軸
plt.show()    
    
    
    
    