package cnn;

import java.io.File;
import java.io.IOException;

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
	
	public DoubleMatrix train(DoubleMatrix input, DoubleMatrix labels, int iterations) throws IOException {
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

    public void fineTune(DoubleMatrix input, DoubleMatrix labels, int iterations, double alpha) throws IOException {
        DoubleMatrix[] results = new DoubleMatrix[layers.length-1];
        for(int i = 0; i < iterations; i++) {
            DoubleMatrix y = labels;
            int j = 0;
            for(; j < layers.length-1; j++) {
                results[j] = layers[j].feedForward(input);
            }
            j++;
            for(; j >=0; j--) {
                //alpha *= 0.96;
                y = layers[j].backPropagation(results, j, y, ((double)i)/iterations, alpha);
            }
        }
    }
	
	public int[][] compute(DoubleMatrix input) {
		for(int i = startLayer; i < layers.length; i++) {
			input = layers[i].compute(input);
		}
		return Utils.computeResults(input);
	}
	
	public DoubleMatrix computeRes(DoubleMatrix input) {
		for(int i = startLayer; i < layers.length; i++) {
			input = layers[i].compute(input);
		}
		return input;
	}
	
}
