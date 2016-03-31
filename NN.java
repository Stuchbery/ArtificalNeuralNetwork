/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author root
 */
public class NN 
{
    private Random rand;
    private Matrix M;
    double[][] yHat;
    double[][] X;
    double[][] Y;
    double[][] Z3;
    double[][] A2;
    double[][] W1;
    double[][] W2;
    double[][] Z2;
    
    public NN()
    {
        M = new Matrix();
        rand = new Random();
        rand.setSeed(System.nanoTime());
        
        //X->W1->Z2->SIG->A2->W2->Z3->SIG->yHat
        //PART ONE 
        this.X = normaliseDataX(initialseDataX());
        this.Y = normaliseDataY(initialseDataY());
        
        //PART TWO
        //defining the neural netowrk size
        int inputLayerSize=2;
        int outputLayerSize=1;
        int hiddenLayerSize=3;
        
        //Create random weights
        this.W1 = setRandWeights(inputLayerSize, hiddenLayerSize);
        this.W2 = setRandWeights(hiddenLayerSize, outputLayerSize);
        
        //Propagate inputs though network
        double cost1 = costFunction();
        while(cost1>0.00000001)
        {
            propagateForward();
            displayDiamensions();
            printMatrixArray(Y,"Y");
            printMatrixArray(yHat,"yHat");

            cost1 = costFunction();
            System.out.println("Minimise this: " + cost1);

            double learningRate=5;
            minimiseCostFunction(learningRate);
            double cost2 = costFunction();
            System.out.println("New cost : " + cost2);
        }
    }
      private void displayDiamensions()
    {
        System.out.println("X:W1:Z2:A2:W2:Z3:yHat");
        System.out.println("["+X.length+"]"+"["+X[0].length+"]  "+"["+W1.length+"]"+"["+W1[0].length+"]  "+"["+Z2.length+"]"+"["+Z2[0].length+"]  "+"["+A2.length+"]"+"["+A2[0].length+"]  "+"["+W2.length+"]"+"["+W2[0].length+"]  "+"["+Z3.length+"]"+"["+Z3[0].length+"]  "+"["+yHat.length+"]"+"["+yHat[0].length+"]  ");
    }
    public static void main(String[] args) 
    {
        NN ann = new NN();
    }

    private double [][] initialseDataX()    //[3][2]        //rows cols
    {
        double[][] buff = new double[3][2];
        buff[0][0]=3;
        buff[0][1]=5;
        
        buff[1][0]=5;
        buff[1][1]=1;
        
        buff[2][0]=10;
        buff[2][1]=2;
        
        return buff;
    }
    private double [][] initialseDataY() 
    {
        double[][] buff = new double[3][1];
        buff[0][0]=75;
        buff[1][0]=82;
        buff[2][0]=93;
        
        return buff;
    }
    private double[][] normaliseDataX(double[][] X) 
    {
//        # Normalize
//        X = X/np.amax(X, axis=0)
//        y = y/100 #Max test score is 100
        double max = findMaxMatrixArray(X);
        return scalarMatrixDivide(X,max);
    }
    private double[][] normaliseDataY(double[][] Y) 
    {
//        # Normalize
//        X = X/np.amax(X, axis=0)
//        y = y/100 #Max test score is 100
        
        //double max = findMaxMatrixArray(y);
        
        return scalarMatrixDivide(Y,100);
    }
    private double[][] setRandWeights(int numOfRows, int numOfCols)
    {
        //0-1
        double[][] newMatrix = new double[numOfRows][numOfCols];
        double max=1;
        double min=0;
        //double[][] buff= new double[in.length][in[0].length];
        
        for(int countRow=0; countRow<newMatrix.length; countRow++)
        {
            for(int countCol=0; countCol<newMatrix[countRow].length; countCol++)
            {
                newMatrix[countRow][countCol]= randomInRange( min, max);
            }
        }
        return newMatrix;
    }
    public double randomInRange(double min, double max) 
    {
      double range = max - min;
      double scaled = rand.nextDouble() * range;
      double shifted = scaled + min;
      return shifted; // == (rand.nextDouble() * (max-min)) + min;
    }
    
    //Propagate Forward
    private void propagateForward()
    {
        //Propagate inputs though network
        this.Z2 = M.dot(X,W1);
        this.A2 = sigmoidMulti(Z2);
        this.Z3 = M.dot(A2,W2);
        this.yHat = sigmoidMulti(Z3);
    }
    //SIGMOID
    private double[][] sigmoidMulti(double[][] in)
    {
        //return 1/(1+np.exp(-z))
        
        double[][] buff= new double[in.length][in[0].length];
        
        for(int countRow=0; countRow<in.length; countRow++)
        {
            for(int countCol=0; countCol<in[countRow].length; countCol++)
            {
                buff[countRow][countCol]=1/(1+Math.exp(-in[countRow][countCol]));
            }
        }
        return buff;
    }
    private double[] sigmoidVector(double[] in)
    {
        //return 1/(1+np.exp(-z))
        
        double[]buff= new double[in.length];
        for(int countCol=0; countCol<in.length; countCol++)
        {
            buff[countCol]=1/(1+Math.exp(-in[countCol]));
        }

        return buff;
    }
    
    //SIGMOID PRIME
    private double[][] sigmoidPrimeMulti(double[][] in)
    {
        //return 1/(1+np.exp(-z))
        
        double[][] buff= new double[in.length][in[0].length];
        
        for(int countRow=0; countRow<in.length; countRow++)
        {
            for(int countCol=0; countCol<in[countRow].length; countCol++)
            {
                buff[countRow][countCol]=Math.exp(-in[countRow][countCol])/(Math.pow((1+Math.exp(-in[countRow][countCol])),2));
                //buff[countRow][countCol]=1/(1+Math.exp(-in[countRow][countCol]));
            }
        }
        return buff;
    }
    private double[] sigmoidPrimeVector(double[] in)
    {
        //return 1/(1+np.exp(-z))
        
        double[]buff= new double[in.length];
        for(int countCol=0; countCol<in.length; countCol++)
        {
            buff[countCol]=Math.exp(-in[countCol])/(Math.pow((1+Math.exp(-in[countCol])),2));
                
        }

        return buff;
    }
    
    //CostFunction
    private double costFunction()   //vectors [rows][0]
    {
        propagateForward();
        double J = 0;
        for(int count=0; count<yHat.length; count++)
        {
            J = J + Math.pow((Y[count][0]-yHat[count][0]),2);
        }
        return J*0.5;
    }
    //deltaJs[0]=dJdW1;
    //deltaJs[1]=dJdW2;
    
    private double[][][] costFunctionPrime(double[][] Y, double[][] X)   //vectors [rows][0]
    {
        //M.multiply performs element wize operations
        //M.dot performs matrix multiplication
        double[][] delta3 = M.multiply(scalarMatrixMulti( M.subtract(Y,this.yHat),-1),sigmoidPrimeMulti(this.Z3));
//        double[][] dJdW2 = M.multiply(M.transpose(this.A2),delta3);
        double[][] dJdW2 = M.dot(M.transpose(this.A2),delta3);
        
        double[][] delta2 = M.multiply(M.dot(delta3,M.transpose(this.W2)),sigmoidPrimeMulti(this.Z2));
        double[][] dJdW1 = M.dot(M.transpose(X),delta2);
        
        double[][][] deltaJs= new double[2][][];
        deltaJs[0]=dJdW1;
        deltaJs[1]=dJdW2;
        return deltaJs;
    }
    
    //Train neural network
    private double minimiseCostFunction(double learningRate)
    {
        double[][][]deltaJs=costFunctionPrime(Y,X);
        this.W1=M.subtract(this.W1, scalarMatrixMulti(deltaJs[0],learningRate));
        this.W2=M.subtract(this.W2, scalarMatrixMulti(deltaJs[1],learningRate));
        
        return costFunction();
    }
    

    //Matrix functions
    private void printMatrixArray(double [][] in)
    {
        for(int countRow=0; countRow<in.length; countRow++)
        {
            for(int countCol=0; countCol<in[countRow].length; countCol++)
            {
                System.out.print("["+in[countRow][countCol]+"]");
            }
            System.out.println();
        }
    }
    private void printMatrixArray(double [][] in,String label)
    {
        System.out.println(label);
        for(int countRow=0; countRow<in.length; countRow++)
        {
            for(int countCol=0; countCol<in[countRow].length; countCol++)
            {
                System.out.print("["+in[countRow][countCol]+"]");
            }
            System.out.println();
        }
    }
    private double findMaxMatrixArray(double [][] in)
    {
        double max=0;
        for(int countRow=0; countRow<in.length; countRow++)
        {
            for(int countCol=0; countCol<in[countRow].length; countCol++)
            {
                if(max<in[countRow][countCol])
                {
                    //new max val
                    max=in[countRow][countCol];
                }
            }
            //System.out.println();
        }
        return max;
    }
    private double[][] scalarMatrixMulti(double[][] in,double multi)
    {
        double[][] buff= new double[in.length][in[0].length];
        
        for(int countRow=0; countRow<in.length; countRow++)
        {
            for(int countCol=0; countCol<in[countRow].length; countCol++)
            {
                buff[countRow][countCol]=multi*in[countRow][countCol];
            }
        }
        return buff;
    }
    private double[][] scalarMatrixDivide(double[][] in,double div)
    {
        double[][] buff= new double[in.length][in[0].length];
        
        for(int countRow=0; countRow<in.length; countRow++)
        {
            for(int countCol=0; countCol<in[countRow].length; countCol++)
            {
                buff[countRow][countCol]=in[countRow][countCol]/div;
            }
        }
        return buff;
    }

}
