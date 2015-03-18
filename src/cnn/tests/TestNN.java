package cnn.tests;

import cnn.NeuralNetwork;
import cnn.SoftmaxClassifier;
import numerical.LBFGS.ExceptionWithIflag;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import java.io.IOException;

public class TestNN {

	@Test
	public void test() throws IOException, ExceptionWithIflag {
		/*SAEImageHandler handler = new SAEImageHandler("trainData", 28*28, 15298);
		DoubleMatrix input = handler.getImages();
		handler = new SAEImageHandler("unlabeledData", 28*28, 29404);
		DoubleMatrix unlabeledData = handler.getImages();
		SAELabelHandler labelHandler = new SAELabelHandler("trainLabels", 15298, 5);
		DoubleMatrix labels = labelHandler.getLabels();
		int hiddenSize = 200;
		double sparsityParam = 0.1;
		double lambda = 3e-3;
		double beta = 3;
		double alpha = 0.3;
		SparseAutoencoder sae = new SparseAutoencoder(unlabeledData.columns, hiddenSize, unlabeledData.columns, sparsityParam, lambda, beta, alpha);
		lambda = 1e-4;
		SoftmaxClassifier sc = new SoftmaxClassifier(lambda, alpha);
		NeuralNetwork nn = new NeuralNetwork(sae, sc);
		nn.train(input, labels, unlabeledData, 200);
		int[] result = nn.compute(input);
		compareResults(result, labels);
		handler = new SAEImageHandler("testData",28*28,15298);
		DoubleMatrix testData = handler.getImages();
		labelHandler = new SAELabelHandler("testLabels",15298,5);
		DoubleMatrix testLabels = labelHandler.getLabels();
		int[] testResult = nn.compute(testData);
		compareResults(testResult, testLabels);*/
	}
	
	public void compareResults(int[] result, DoubleMatrix labels) {
		double sum = 0;
		for(int i = 0; i < result.length; i++) {
			sum += labels.get(i, result[i]);
		}
		System.out.println(sum+"/"+result.length);
		System.out.println(sum/result.length);
	}

}
