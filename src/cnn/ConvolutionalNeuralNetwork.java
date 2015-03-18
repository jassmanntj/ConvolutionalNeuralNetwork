package cnn;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicLong;

import org.jblas.DoubleMatrix;

public class ConvolutionalNeuralNetwork {
	private NeuralNetworkLayer[] layers;
	private int startLayer;
	private String name;
	
	public ConvolutionalNeuralNetwork(NeuralNetworkLayer[] layers, String name) {
		this.layers = layers;
		this.name = name;
		startLayer = 0;
		for(int i = 0; i < layers.length; i++) {
			if(layers[i] instanceof ConvolutionLayer) startLayer = i;
		}
	}
	
	public DataContainer train(DataContainer input, DoubleMatrix labels, int iterations) throws IOException {
		for(int i = 0; i < layers.length; i++) {
			File f = new File(name+"Layer"+i+".csv");
			if(f.exists() || new File(0+name+"Layer"+i+".layer").exists()) {
				System.out.println("ALayer"+i);
				layers[i].loadLayer(name+"Layer"+i+".layer");
				input = layers[i].compute(input);
			}
			else {
				System.out.println("BLayer"+i);
				input = layers[i].train(input, labels, iterations);
				layers[i].writeTheta(name+"Layer"+i+".csv");
				layers[i].writeLayer(name+"Layer"+i+".layer");
			}
		}
		return input;
	}

    public void fineTune(DataContainer in, DoubleMatrix labels, int iterations, double alpha, int batchSize) throws IOException {
        DataContainer[] results = new DataContainer[layers.length];
        DoubleMatrix[][] input = in.getDataArray();
        for(int i = 0; i < iterations; i++) {
            AtomicLong cost = new AtomicLong(0);
            for(int k = 0; k < (in.length() + batchSize-1)/batchSize; k++) {
                DoubleMatrix y = labels.getRange(k * batchSize, k * batchSize + batchSize, 0, labels.columns);
                DoubleMatrix[][] x = new DoubleMatrix[batchSize][in.channels()];
                for(int z = 0; z < batchSize; z++) {
                    x[z] = input[k*batchSize+z];
                }
                DataContainer xc = new DataContainer(x);
                results[0] = xc;
                int j = 1;
                for (; j < layers.length; j++) {
                    xc = layers[j-1].feedForward(xc);
                    results[j] = xc;
                }
                for (; j >= 1; j--) {
                    //alpha *= 0.96;
                    y = layers[j-1].backPropagation(results, j, y, /*((double) i) / iterations*/0.9, alpha);
                    if (j == layers.length) cost.getAndAdd(Double.doubleToLongBits(layers[j-1].getCost()));
                }
                System.out.println("Cost "+k+": "+Double.longBitsToDouble(cost.get()));
            }
            System.out.println("Cost: "+Double.longBitsToDouble(cost.get()));
        }
    }
	
	public int[][] compute(DataContainer input) {
		for(int i = startLayer; i < layers.length; i++) {
			input = layers[i].compute(input);
		}
		return Utils.computeResults(input.getDataArray());
	}
	
	public DataContainer computeRes(DataContainer input) {
		for(int i = startLayer; i < layers.length; i++) {
			input = layers[i].compute(input);
		}
		return input;
	}
	
}
