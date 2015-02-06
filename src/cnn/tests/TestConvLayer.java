package cnn.tests;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cnn.ConvolutionLayer;
import cnn.device.DeviceConvolutionLayer;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class TestConvLayer {

	@Test
	public void test() {
		DoubleMatrix currentImage = new DoubleMatrix(1, 60*80*3);
		for(int i = 0; i < currentImage.columns; i++) {
			currentImage.put(0,i, 1.0/64);
		}
		
		DenseDoubleMatrix2D ci = new DenseDoubleMatrix2D(currentImage.toArray2());
		
		ConvolutionLayer c = new ConvolutionLayer("LeafCNNaLayer1.layer");
		DeviceConvolutionLayer c2 = new DeviceConvolutionLayer("LeafCNNaLayer1.layer");
		
		DoubleMatrix res1 = c.compute(currentImage);
		DoubleMatrix2D res2 = c2.compute(ci);
		
		System.out.println(res1);
		System.out.println(res2);
	}

}
