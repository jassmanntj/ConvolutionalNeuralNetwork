package cnn.tests;

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cnn.Utils;
import cnn.device.DeviceUtils;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import java.io.IOException;

public class TestConv2d {

	@Test
	public void test() throws IOException {
		DoubleMatrix input = DoubleMatrix.rand(7,7);
		DoubleMatrix kernel = DoubleMatrix.ones(2, 2);
		//for(int i = 0; i < kernel.rows; i++) {
		//	System.out.println(kernel.getRow(i));
		//}
		DoubleMatrix result = Utils.conv2d(input, kernel);
		
		DenseDoubleMatrix2D in = new DenseDoubleMatrix2D(input.toArray2());
		DenseDoubleMatrix2D ker = new DenseDoubleMatrix2D(kernel.toArray2());
		//System.out.println(ker);
		DenseDoubleMatrix2D res = DeviceUtils.conv2d(in, ker);
		
		
		//System.out.println(input);
		//System.out.println(kernel);
		for(int i = 0; i < result.rows; i++) {
			System.out.println(result.getRow(i));
		}
		//System.out.println(result);
		
		System.out.println("--------------------------");
		//System.out.println(in);
		//System.out.println(ker);
		System.out.println(res);
	}


}
