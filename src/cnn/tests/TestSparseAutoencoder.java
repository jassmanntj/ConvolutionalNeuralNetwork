package cnn.tests;/*package cnnTests;

import java.io.IOException;

import numerical.LBFGS.ExceptionWithIflag;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import cnn.ImageHandler;
import cnn.SparseAutoencoder;

public class TestSparseAutoencoder {

	@Test
	public void test() throws IOException, ExceptionWithIflag {
		ImageHandler handler = new ImageHandler("images.csv", 512, 512);
		DoubleMatrix input = handler.sample(8, 100);
		SparseAutoencoder ae = new SparseAutoencoder(8*8, 25, 8*8, .01, .0001, 3, .5);
		ae.gradientDescent(input, input, 25);
		//ae.lbfgsTrain(input, input, 200);
		//ae.writeTheta("a.csv");
		ae.visualize(8);
	}

}*/

