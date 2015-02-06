package cnn.tests;

import cnn.Utils;
import numerical.LBFGS.ExceptionWithIflag;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class TestZCAWhite {

	@Test
	public void test() throws IOException, ExceptionWithIflag {
		SAEImageHandler handler = new SAEImageHandler("zcaimages.csv", 12*12, 10000);
		DoubleMatrix input = handler.getImages();
		//Utils.visualize(12, 6, input, "ZCApreimages.png");
		DoubleMatrix meanPatch = input.rowMeans();
		DoubleMatrix ZCAWhite = Utils.calculateZCAWhite(input, meanPatch, 0.1);
		input = Utils.ZCAWhiten(input, meanPatch, ZCAWhite);
		FileWriter fw = new FileWriter("ZCAimagess.csv");
		BufferedWriter writer = new BufferedWriter(fw);
		double[][] inArr = input.toArray2();
		for(double[] d : inArr) {
			for(int i = 0; i < d.length; i++) {
				if(i > 0) writer.write(",");
				writer.write(""+(d[i]));
			}
			writer.write('\n');
		}
		writer.close();
		//Utils.visualize(12, 6, input, "ZCApostimages.png");
		
	}

}