package cnn;

import java.io.IOException;

import org.jblas.DoubleMatrix;

public abstract class NeuralNetworkLayer {
	public abstract DoubleMatrix compute(DoubleMatrix input);
	public abstract DoubleMatrix getTheta();
	public abstract DoubleMatrix getBias();
	public abstract void writeTheta(String filename);
	public abstract DoubleMatrix loadTheta(String filename, DoubleMatrix input);
	public abstract DoubleMatrix train(DoubleMatrix input, DoubleMatrix output, int iterations) throws IOException;
	public abstract void writeLayer(String filename);
	public abstract void loadLayer(String filename);
    public abstract DoubleMatrix feedForward(DoubleMatrix input);
    public abstract DoubleMatrix backPropagation(DoubleMatrix[] results, int layer, DoubleMatrix y, double momentum, double alpha);
    public abstract DoubleMatrix getA();
    public double getCost() {
        return 0;
    }
}
