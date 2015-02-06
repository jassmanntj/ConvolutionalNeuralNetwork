package cnn.tests;

import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class SAEImageHandler {
	private double[] images;
	private int size;
	private int imgCount;
	public SAEImageHandler(String fileLocation, int size, int imgCount) throws IOException {
		this.size = size;
		this.imgCount = imgCount;
		images = new double[imgCount*size];
		@SuppressWarnings("resource")
		BufferedReader br = new BufferedReader(new FileReader(fileLocation));
		String line = br.readLine();
		for(int iCnt = 0; iCnt < size; iCnt++) {
			if(line==null) System.out.println(iCnt);
			String[] imageStr = line.split(",");
			for(int i = 0; i < imgCount; i++) {
				images[iCnt*imgCount+i] = Double.parseDouble(imageStr[i]);
			}
			line = br.readLine();
		}
	}
	
	public DoubleMatrix getImages() {
		DoubleMatrix result = DoubleMatrix.zeros(imgCount, size);
		result.data = images;
		return result;
		
	}
}
