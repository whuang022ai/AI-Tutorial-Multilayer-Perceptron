/***************************************************************************
*
*  main.cpp
*  
*
*
* This is a hard coding version of Sigmoid Three Layer Neural Network. 
* 2 input *2 hideen *1 output.
* It is just for "Hello World" for the A.I beginners.
*
*
***************************************************************************
* Copyright 2018 whuang022.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* 
 **************************************************************************/
 
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>

double eta=0.3;//learning rate
//to h1
double w1=0;
double w2=0;
//to h2
double w3=0;
double w4=0;
//to o1
double w5=0;
double w6=0;
//h1 b
double b1=0;
//h2 b
double b2=0;
//o1
double b3=0;

// The Neural 2*2*1
// neural(input1,input2,desire output)
double neural(double x1,double x2,double d){
    //forward path
    //input layer -> hidden layer
    double n1=w1*x1+w2*x2+b1;
    double n2=w3*x1+w4*x2+b2;
    double h1 =1/(1+exp(-1*n1));
    double h2 =1/(1+exp(-1*n2));
    //hidden layer -> output layer
    double n3=w5*h1+w6*h2+b3;
    double o = 1/(1+exp(-1*n3));
    //backward path
    //output layer->hidden layer 
    w5+=eta*(d-o)*o*(1-o)*h1;
    w6+=eta*(d-o)*o*(1-o)*h2;
    b3+=eta*(d-o)*o*(1-o)* 1;
    //hidden layer->input layer 
    w1+=eta*(d-o)*o*(1-o)*w5*(h1)*(1-h1)*x1;
    w2+=eta*(d-o)*o*(1-o)*w5*(h1)*(1-h1)*x2; 
    w3+=eta*(d-o)*o*(1-o)*w6*(h2)*(1-h2)*x1;
    w4+=eta*(d-o)*o*(1-o)*w6*(h2)*(1-h2)*x2;   
	b1+=eta*(d-o)*o*(1-o)*w5*(h1)*(1-h1)* 1; 
	b2+=eta*(d-o)*o*(1-o)*w6*(h2)*(1-h2)* 1; 
    return (d-o);
}

// learn AND
// this will converge
void neuralAND(){
   	int	 x=rand()%2;
	int	 y=rand()%2;
	int  z=x&y;
	std::cout << neural(x,y,z)<<"\n";
}

// learn OR
// this will converge
void neuralOR(){
	int	 x=rand()%2;
	int	 y=rand()%2;
	int  z=x|y;
	std::cout << neural(x,y,z)<<"\n";
}

// learn XOR
// this will converge
void neuralXOR(){
	int	 x=rand()%2;
	int	 y=rand()%2;
	int  z=x^y;
	std::cout << neural(x,y,z)<<"\n";
}

// Init the weight
// random 0~1 
void init(){
    srand(time(0));
	w1=double(std::rand()) / (double(RAND_MAX) + 1.0);
	w2=double(std::rand()) / (double(RAND_MAX) + 1.0);
	w3=double(std::rand()) / (double(RAND_MAX) + 1.0);
	w4=double(std::rand()) / (double(RAND_MAX) + 1.0);
	w5=double(std::rand()) / (double(RAND_MAX) + 1.0);
	w6=double(std::rand()) / (double(RAND_MAX) + 1.0);
	b1=double(std::rand()) / (double(RAND_MAX) + 1.0);
	b2=double(std::rand()) / (double(RAND_MAX) + 1.0);
	b3=double(std::rand()) / (double(RAND_MAX) + 1.0);
}

int main(int argc, char** argv) {
	init();
	for(int i=0;i<300000;i++){
		neuralXOR();
	}
}
