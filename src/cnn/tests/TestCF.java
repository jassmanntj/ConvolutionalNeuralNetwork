package cnn.tests;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cnn.ConvolutionLayer;
import cnn.device.DeviceConvolutionLayer;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import java.io.IOException;

public class TestCF {

	@Test
	public void test() throws IOException {
		DoubleMatrix currentImage = DoubleMatrix.rand(1, 60*80*3);
		
		DenseDoubleMatrix2D ci = new DenseDoubleMatrix2D(currentImage.toArray2());
		
		ConvolutionLayer c = new ConvolutionLayer("LeafCNNaLayer1.layer");
		DeviceConvolutionLayer c2 = new DeviceConvolutionLayer("LeafCNNaLayer1.layer");
		
		DoubleMatrix res1 = c.convFeature(currentImage, 0);
		DoubleMatrix2D res2 = c2.convFeature(ci, 0);
		
		System.out.println(res1);
		System.out.println(res2);
	}

}
