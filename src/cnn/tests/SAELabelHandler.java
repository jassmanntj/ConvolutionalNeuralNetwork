package cnn.tests;

import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class SAELabelHandler {
	private double[][] labels;
	public SAELabelHandler(String fileLocation, int imgCount, int classes) throws IOException {
		@SuppressWarnings("resource")
		BufferedReader br = new BufferedReader(new FileReader(fileLocation));
		String line = br.readLine();
		labels = new double[imgCount][classes];
		for(int iCnt = 0; iCnt < imgCount; iCnt++) {
			int label = Integer.parseInt(line);
			for(int i = 0; i < classes; i++) {
				labels[iCnt][i] = i==label?1:0;
			}
			line = br.readLine();
		}
	}
	
	public DoubleMatrix getLabels() {
		return new DoubleMatrix(labels);
	}
}
