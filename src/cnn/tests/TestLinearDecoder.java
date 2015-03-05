package cnn.tests;

import cnn.LinearDecoder;
import cnn.Utils;
import numerical.LBFGS.ExceptionWithIflag;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import java.io.IOException;

public class TestLinearDecoder {

	@Test
	public void test() throws IOException, ExceptionWithIflag {
		/*SAEImageHandler handler = new SAEImageHandler("patches.csv", 8*8*3, 100000);
		DoubleMatrix input = handler.getImages();
		//scale data
		input.divi(input.max());
		DoubleMatrix meanPatch = input.columnMeans();
		DoubleMatrix ZCAWhite = Utils.calculateZCAWhite(input, meanPatch, 0.1);
		input = Utils.ZCAWhiten(input, meanPatch, ZCAWhite);
		LinearDecoder ae = new LinearDecoder(8, 3, 400, .035, 3e-3, 5, .5);
		//ae.gradientDescent(input, input, 400);
		ae.lbfgsTrain(input, 400);
		System.out.println(input.rows+":"+input.columns);
		ae.visualize(8,20, "Features.png");*/
	}
}