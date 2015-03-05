package cnn.tests;


import cnn.DeepNN;
import cnn.SoftmaxClassifier;
import cnn.SparseAutoencoder;
import cnn.Utils;
import numerical.LBFGS.ExceptionWithIflag;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import java.io.IOException;

public class TestDeepNN {

	@Test
	public void test() throws IOException, ExceptionWithIflag {
		/*SAEImageHandler handler = new SAEImageHandler("train-data.csv", 28*28, 60000);
		DoubleMatrix input = handler.getImages();
		SAELabelHandler labelHandler = new SAELabelHandler("train-labels.csv", 60000, 10);
		DoubleMatrix labels = labelHandler.getLabels();
		int hiddenSize1 = 200;
		int hiddenSize2 = 200;
		double sparsityParam = 0.1;
		double lambda = 3e-3;
		double beta = 3;
		double alpha = 0.3;
		SparseAutoencoder[] saes = new SparseAutoencoder[2];
		saes[0] = new SparseAutoencoder(input.columns, hiddenSize1, input.columns, sparsityParam, lambda, beta, alpha);
		saes[1] = new SparseAutoencoder(hiddenSize1, hiddenSize2, hiddenSize1, sparsityParam, lambda, beta, alpha);
		lambda = 1e-4;
		SoftmaxClassifier sc = new SoftmaxClassifier(lambda, alpha);
		DeepNN nn = new DeepNN(saes, sc);
		nn.train(input, labels, 400);
		int[][] result = Utils.computeResults(nn.compute(input));
		compareResults(result, labels);
		handler = new SAEImageHandler("test-data.csv",28*28,10000);
		DoubleMatrix testData = handler.getImages();
		labelHandler = new SAELabelHandler("test-labels.csv",10000,10);
		DoubleMatrix testLabels = labelHandler.getLabels();
		int[][] testResult = Utils.computeResults(nn.compute(testData));
		compareResults(testResult, testLabels);*/
	}
	
	public void compareResults(int[][] result, DoubleMatrix labels) {
		double sum = 0;
		for(int i = 0; i < result.length; i++) {
			sum += labels.get(i, result[i][0]);
		}
		System.out.println(sum+"/"+result.length);
		System.out.println(sum/result.length);
	}

}