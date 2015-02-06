package cnn.tests;

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cnn.Utils;
import cnn.device.DeviceUtils;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class TestSigmoid {

	@Test
	public void test() {
		DoubleMatrix input = new DoubleMatrix(2, 2);
		input.put(0, 0, 1);
		input.put(0, 1, 2);
		input.put(1, 0, 3);
		input.put(1, 1, 4);
		DenseDoubleMatrix2D in = new DenseDoubleMatrix2D(input.toArray2());
		
		DoubleMatrix result = Utils.sigmoid(input);
		DenseDoubleMatrix2D res = DeviceUtils.sigmoid(in);
		
		System.out.println(result);
		System.out.println(res);
	}

}
