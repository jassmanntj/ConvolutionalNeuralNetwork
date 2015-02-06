package cnn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.jblas.DoubleMatrix;

public class ConvolutionLayer extends NeuralNetworkLayer {
	private int channels;
	private int patchDim;
	private int imageRows;
	private int imageCols;
	private int poolDim;
	private int numPatches;
	private double sparsityParam;
	private double lambda;
	private double beta;
	private double alpha;
	private DoubleMatrix whitenedTheta;
	private DoubleMatrix whitenedBias;
	private DoubleMatrix pooledFeatures;
	private int patchSize;
	private int imageSize;
	private int resultRows;
	private int resultCols;
	private boolean whiten;
	private DoubleMatrix images;
	private static final int NUMTHREADS = 8;
	
	public ConvolutionLayer(int channels, int patchDim, int imageRows, int imageCols, int poolDim, int numPatches, double sparsityParam, double lambda, double beta, double alpha, boolean whiten) {
		this.channels = channels;
		this.patchDim = patchDim;
		this.imageRows = imageRows;
		this.imageCols = imageCols;
		this.poolDim = poolDim;
		this.numPatches = numPatches;
		this.sparsityParam = sparsityParam;
		this.lambda = lambda;
		this.beta = beta;
		this.alpha = alpha;
		this.whiten = whiten;
	}
	
	public ConvolutionLayer(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] data = reader.readLine().split(",");
			whitenedTheta = new DoubleMatrix(Integer.parseInt(data[0]),Integer.parseInt(data[1]));
			data = reader.readLine().split(",");
			for(int i = 0; i < whitenedTheta.rows; i++) {
				for(int j = 0; j < whitenedTheta.columns; j++) {
					whitenedTheta.put(i, j, Double.parseDouble(data[i*whitenedTheta.columns+j]));
				}
			}
			data = reader.readLine().split(",");
			whitenedBias = new DoubleMatrix(Integer.parseInt(data[0]), Integer.parseInt(data[1]));
			data = reader.readLine().split(",");
			for(int i = 0; i < whitenedBias.rows; i++) {
				for(int j = 0; j < whitenedBias.columns; j++) {
					whitenedBias.put(i, j, Double.parseDouble(data[i*whitenedBias.columns+j]));
				}
			}
			imageRows = Integer.parseInt(reader.readLine());
			imageCols = Integer.parseInt(reader.readLine());
			patchDim = Integer.parseInt(reader.readLine());
			poolDim = Integer.parseInt(reader.readLine());
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public int getOutputSize() {
		return numPatches * ((imageRows-patchDim+1)/poolDim) * ((imageCols - patchDim+1)/poolDim);
	}
	
	private DoubleMatrix convolve() {
		System.out.println("Starting Convolution");
		int numFeatures = whitenedTheta.columns;
		this.resultRows = (imageRows-patchDim+1)/poolDim;
		this.resultCols = (imageCols-patchDim+1)/poolDim;
		this.patchSize = patchDim*patchDim;
		this.imageSize = imageRows*imageCols;
		this.pooledFeatures = new DoubleMatrix(images.rows, numFeatures * (resultRows * resultCols));
		ExecutorService executor = Executors.newFixedThreadPool(NUMTHREADS);
		for(int imageNum = 0; imageNum < images.rows; imageNum++) {
			Runnable worker = new ConvolutionThread(imageNum);
			executor.execute(worker);
		}
		executor.shutdown();
		while(!executor.isTerminated());
		return pooledFeatures;
	}
	
	private class ConvolutionThread implements Runnable {
		private int imageNum;

		public ConvolutionThread(int imageNum) {
			this.imageNum = imageNum;
		}

		@Override
		public void run() {
			System.out.println("Image: " + imageNum);
			DoubleMatrix currentImage = images.getRow(imageNum);
			for (int featureNum = 0; featureNum < whitenedTheta.columns; featureNum++) {
				DoubleMatrix convolvedFeature = null;
				try {
					convolvedFeature = convFeature(currentImage, featureNum);
				} catch (IOException e) {
					e.printStackTrace();
				}
				pool(convolvedFeature, imageNum, featureNum);
			}
		}

	}
	
	public DoubleMatrix convFeature(DoubleMatrix currentImage, int featureNum) throws IOException {
		DoubleMatrix convolvedFeature = DoubleMatrix.zeros(imageRows-patchDim+1,imageCols - patchDim+1);
		for(int channel = 0; channel < channels; channel++) {
			DoubleMatrix feature = whitenedTheta.getRange(patchSize*channel, patchSize*channel+patchSize,featureNum, featureNum+1);
			feature.reshape(patchDim, patchDim);
			DoubleMatrix image = currentImage.getRange(0, 1, imageSize*channel,imageSize*channel+imageSize);
			image.reshape(imageRows, imageCols);
			DoubleMatrix conv = Utils.conv2d(image, feature);
			convolvedFeature.addi(conv);
		}
		return Utils.sigmoid(convolvedFeature.add(whitenedBias.get(featureNum)));
	}
	
	public void pool(DoubleMatrix convolvedFeature, int imageNum, int featureNum) {
		for(int poolRow = 0; poolRow < resultRows; poolRow++) {
			for(int poolCol = 0; poolCol < resultCols; poolCol++) {
				DoubleMatrix patch = convolvedFeature.getRange(poolRow*poolDim, poolRow*poolDim+poolDim, poolCol*poolDim, poolCol*poolDim+poolDim);
				pooledFeatures.put(imageNum, featureNum*resultRows*resultCols+poolRow*resultCols+poolCol, patch.mean());
			}
		}
	}
	
	public void writeTheta(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			for(int i = 0; i < pooledFeatures.length; i++){
				if( i < pooledFeatures.length-1)
					writer.write(pooledFeatures.data[i]+",");
				else writer.write(""+pooledFeatures.data[i]);
			}
			writer.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public DoubleMatrix loadTheta(String filename, DoubleMatrix input) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			pooledFeatures = new DoubleMatrix((imageRows-patchDim+1)/poolDim,(imageCols-patchDim+1)/poolDim);
			String[] data = reader.readLine().split(",");
			for(int i = 0; i < pooledFeatures.data.length; i++) {
				pooledFeatures.data[i] = Double.parseDouble(data[i]);
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		return pooledFeatures;
	}

	public void loadLayer(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] line = reader.readLine().split(",");
			whitenedTheta = new DoubleMatrix(Integer.parseInt(line[0]), Integer.parseInt(line[1]));
			line = reader.readLine().split(",");
			for(int i = 0; i < whitenedTheta.rows; i++) {
				for(int j = 0; j < whitenedTheta.columns; j++) {
					whitenedTheta.put(i, j, Double.parseDouble(line[i * whitenedTheta.columns + j]));
				}
			}
			line = reader.readLine().split(",");
			for(int i = 0; i < whitenedBias.rows; i++) {
				for(int j = 0; j < whitenedBias.columns; j++) {
					whitenedBias.put(i, j, Double.parseDouble(line[i * whitenedBias.columns + j]));
				}
			}
			imageRows = Integer.parseInt(reader.readLine());
			imageCols = Integer.parseInt(reader.readLine());
			patchDim = Integer.parseInt(reader.readLine());
			poolDim = Integer.parseInt(reader.readLine());
			reader.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public void writeLayer(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			writer.write(whitenedTheta.rows+","+whitenedTheta.columns+"\n");
			for(int i = 0; i < whitenedTheta.rows; i++) {
				for(int j = 0; j < whitenedTheta.columns; j++) {
					writer.write(whitenedTheta.get(i,j)+",");
				}
			}
			writer.write("\n"+whitenedBias.rows+","+whitenedBias.columns+"\n");
			for(int i = 0; i < whitenedBias.rows; i++) {
				for(int j = 0; j < whitenedBias.columns; j++) {
					writer.write(whitenedBias.get(i,j)+",");
				}
			}
			writer.write("\n"+imageRows+"\n"+imageCols+"\n"+patchDim+"\n"+poolDim);
			writer.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	
	public DoubleMatrix compute(DoubleMatrix input) {
		this.images = input;
		this.pooledFeatures = convolve();
		return this.pooledFeatures;
	}

	@Override
	public DoubleMatrix getTheta() {
		return null;
	}

	@Override
	public DoubleMatrix getBias() {
		return null;
	}

	@Override
	public DoubleMatrix train(DoubleMatrix input, DoubleMatrix output, int iterations) {
		LinearDecoder ae = new LinearDecoder(patchSize, channels, numPatches, sparsityParam, lambda, beta, alpha);
		DoubleMatrix patches = ImageLoader.sample(patchSize, numPatches, input, imageCols, imageRows, channels);
		if(whiten) {
			patches.divi(patches.max());
			DoubleMatrix meanPatch = patches.columnMeans();
			DoubleMatrix ZCAWhite = Utils.calculateZCAWhite(patches, meanPatch, 0.1);
			patches = Utils.ZCAWhiten(patches, meanPatch, ZCAWhite);
			ae.train(patches, patches, iterations);
			DoubleMatrix previousTheta = ae.getTheta();
			DoubleMatrix previousBias = ae.getBias();
			whitenedTheta = ZCAWhite.mmul(previousTheta);
			whitenedBias = previousBias.sub(meanPatch.mmul(whitenedTheta));
		}
		else {
			ae.train(patches, patches, iterations);
			this.whitenedTheta = ae.getTheta();
			this.whitenedBias = ae.getBias();
		}
		return compute(input);
		
	}
}
