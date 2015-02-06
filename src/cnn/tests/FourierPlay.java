package cnn.tests;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cnn.device.DeviceUtils;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class FourierPlay {

	@Test
	public void test() {
		DenseDoubleMatrix2D a = new DenseDoubleMatrix2D(DoubleMatrix.rand(10,10).toArray2());
		DenseDoubleMatrix2D b = new DenseDoubleMatrix2D(DoubleMatrix.rand(2,2).toArray2());
		long t1 = System.nanoTime();
		DoubleMatrix2D resA = DeviceUtils.conv2d(a, b);
		long t2 = System.nanoTime();
		DoubleMatrix2D resB = DeviceUtils.conv2d2(a, b);
		long t3 = System.nanoTime();
		
		System.out.println(resA);
		System.out.println(resB);
		System.out.println((t2-t1)+":"+(t3-t2));
		
		//printMatrix(a);
		//printMatrix(b);
	}
	
	/*public void printMatrix(DoubleMatrix2D mat) {
		for(int i = 0; i < mat.rows(); i++) {
			System.out.println(mat.getRow(i));
		}
		System.out.println("\n");
	}*/

}
