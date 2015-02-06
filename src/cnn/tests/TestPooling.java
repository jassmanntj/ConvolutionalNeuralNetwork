package cnn.tests;

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cnn.ConvolutionLayer;
import cnn.device.DeviceConvolutionLayer;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class TestPooling {

	@Test
	public void test() {
		DoubleMatrix convolvedFeature = new DoubleMatrix(4,4);
		for(int i = 0; i < convolvedFeature.rows; i++) {
			for(int j = 0; j < convolvedFeature.columns; j++) {
				convolvedFeature.put(i,j, i*convolvedFeature.columns+j);
			}
		}
		
		DenseDoubleMatrix2D cf = new DenseDoubleMatrix2D(convolvedFeature.toArray2());
		
		ConvolutionLayer c = new ConvolutionLayer("LeafCNNaLayer1.layer");
		DeviceConvolutionLayer c2 = new DeviceConvolutionLayer("LeafCNNaLayer1.layer");
		
		//DoubleMatrix pooledFeature = c.pool(convolvedFeature, imageNum, featureNum);
	}

}
