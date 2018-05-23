/***************************************************************************
*
*  NeuralNetXor.java
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

package neuralnetxor;

import java.util.Random;

/**
 * NeuralNetXor
 * @author whuang022
 */
public class NeuralNetXor {
    
    private Random ran = new Random();   
    private double eta=0.3;//learning rate
    //to h1
    private double w1=0;
    private double w2=0;
    //to h2
    private double w3=0;
    private double w4=0;
    //to o1
    private double w5=0;
    private double w6=0;
    //h1 b
    private double b1=0;
    //h2 b
    private double b2=0;
    //o1
    private double b3=0;
    
    double boolTdouble(boolean in){
        if(in){
            return 1.0;
        }else{
            return 0.0;
        }
    }
    
    // The Neural 2*2*1
    // neural(input1,input2,desire output)
    private double neural(double x1,double x2,double d){
        //forward path
        //input layer -> hidden layer
        double n1=w1*x1+w2*x2+b1;
        double n2=w3*x1+w4*x2+b2;
        double h1 =1/(1+Math.pow(Math.E,(-1*n1)));
        double h2 =1/(1+Math.pow(Math.E,(-1*n2)));
        //hidden layer -> output layer
        double n3=w5*h1+w6*h2+b3;
        double o =1/(1+Math.pow(Math.E,(-1*n3)));
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
    public void neuralAND(){
       boolean x=ran.nextBoolean();
       boolean y=ran.nextBoolean();
       boolean z=x&y;
       System.out.println(neural(boolTdouble(x),boolTdouble(y),boolTdouble(z)));
    }

    // learn OR
    // this will converge
    public void neuralOR(){
       boolean x=ran.nextBoolean();
       boolean y=ran.nextBoolean();
       boolean z=x|y;
       System.out.println(neural(boolTdouble(x),boolTdouble(y),boolTdouble(z)));
    }

    // learn XOR
    // this will converge
    public void neuralXOR(){
       boolean x=ran.nextBoolean();
       boolean y=ran.nextBoolean();
       boolean z=x^y;
       System.out.println(neural(boolTdouble(x),boolTdouble(y),boolTdouble(z)));
    }

    // Init the weight
    // random 0~1 
    public void init(){
        w1=ran.nextDouble();
        w2=ran.nextDouble();
        b1=ran.nextDouble();
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        NeuralNetXor neural=new NeuralNetXor();
        neural.init();
        for(int i=0;i<300000;i++){
            neural.neuralXOR();
	}
    }
    
}
