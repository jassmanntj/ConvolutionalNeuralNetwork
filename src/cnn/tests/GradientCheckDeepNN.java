package cnn.tests;

import cnn.DeepNN;
import cnn.SoftmaxClassifier;
import cnn.SparseAutoencoder;
import numerical.LBFGS.ExceptionWithIflag;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertTrue;

public class GradientCheckDeepNN {

	@Test
	public void test() throws IOException, ExceptionWithIflag {
		DoubleMatrix input = DoubleMatrix.randn(7, 4);
		double[][] labs = {{0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}};
		DoubleMatrix labels = new DoubleMatrix(labs);
		int hiddenSize1 = 5;
		int hiddenSize2 = 5;
		double sparsityParam = 0.1;
		double lambda = 3e-3;
		double beta = 3;
		double alpha = 0.3;
		SparseAutoencoder[] saes = new SparseAutoencoder[2];
		saes[0] = new SparseAutoencoder(input.columns, hiddenSize1, input.columns, sparsityParam, lambda, beta, alpha);
		saes[1] = new SparseAutoencoder(hiddenSize1, hiddenSize2, hiddenSize1, sparsityParam, lambda, beta, alpha);
		lambda = 1e-4;
		SoftmaxClassifier sc = new SoftmaxClassifier(lambda);	
		DeepNN nn = new DeepNN(saes, sc);
		nn.train(input, labels, 1);
		double diff = nn.gradientChecking(input, labels);
		System.out.println(diff);
		assertTrue( "Diff must be < 1e-9", diff < 1e-9);
	}

}
