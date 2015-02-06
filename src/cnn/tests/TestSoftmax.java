package cnn.tests;

import cnn.SoftmaxClassifier;
import numerical.LBFGS.ExceptionWithIflag;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import java.io.IOException;

public class TestSoftmax {


	@Test
	public void test() throws IOException, ExceptionWithIflag {
		SAEImageHandler handler = new SAEImageHandler("train-data.csv", 8*8, 6000);
		DoubleMatrix input = handler.getImages();
		SAELabelHandler labelHandler = new SAELabelHandler("train-labels.csv", 6000, 10);
		DoubleMatrix labels = labelHandler.getLabels();
		SoftmaxClassifier classifier = new SoftmaxClassifier(0.001);	
		//classifier.lbfgsTrain(input, labels, 50);
		classifier.gradientDescent(input, labels, 10, 0.3);
		int[] result = classifier.computeResults(input);
		//printResults(result, labels);
		compareResults(result, labels);
		handler = new SAEImageHandler("test-data.csv",28*28,10000);
		DoubleMatrix testData = handler.getImages();
		labelHandler = new SAELabelHandler("test-labels.csv",10000,10);
		DoubleMatrix testLabels = labelHandler.getLabels();
		int[] testResult = classifier.computeResults(testData);
		//printResults(testResult, testLabels);
		compareResults(testResult, testLabels);
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
