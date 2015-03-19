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
	private int patchRows;
    private int patchCols;
	private int imageRows;
	private int imageCols;
	private int poolRows;
    private int poolCols;
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
	private DoubleMatrix[][] images;
	
	public ConvolutionLayer(int channels, int patchDim, int imageRows, int imageCols, int poolDim, int numPatches, double sparsityParam, double lambda, double beta, double alpha, boolean whiten) {
		this.channels = channels;
		this.patchRows = patchDim;
        this.patchCols = patchDim;
        this.poolRows = poolDim;
        this.poolCols = poolDim;
		this.imageRows = imageRows;
		this.imageCols = imageCols;
		this.numPatches = numPatches;
		this.sparsityParam = sparsityParam;
		this.lambda = lambda;
		this.beta = beta;
		this.alpha = alpha;
		this.whiten = whiten;
        this.patchSize = patchDim*patchDim;
        double r = Math.sqrt(6)/Math.sqrt(patchRows*patchCols*channels+numPatches+1);
        this.whitenedTheta = DoubleMatrix.rand(patchRows*patchCols*channels, numPatches).muli(2*r).subi(r);
        this.whitenedBias = DoubleMatrix.zeros(1, numPatches);
	}

    public DoubleMatrix getA() {
        return null;
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
            patchRows = Integer.parseInt(reader.readLine());
            patchCols = Integer.parseInt(reader.readLine());
			poolRows = Integer.parseInt(reader.readLine());
            poolCols = Integer.parseInt(reader.readLine());
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}

    public int getOutputRows() {
        return (imageRows-patchRows+1)/poolRows;
    }

    public int getOutputColumns() {
        return (imageCols - patchCols+1)/poolCols;
    }

    private DoubleMatrix[][] oonv(DoubleMatrix[][] input, DoubleMatrix features) {
        System.out.println("Starting Convolution");
        this.imageSize = imageRows*imageCols;
        this.resultRows = (imageRows-patchRows+1);
        this.resultCols = (imageCols-patchCols+1);
        DoubleMatrix[][] convolvedFeatures = new DoubleMatrix[input.length][features.columns];
        //DoubleMatrix convolvedFeatures = new DoubleMatrix(input.rows, features.columns * resultRows * resultCols);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            System.out.println(imageNum);
            DoubleMatrix[] currentImage = input[imageNum];
            for (int featureNum = 0; featureNum < features.columns; featureNum++) {
                DoubleMatrix convolvedFeature = convFeature(currentImage, features, featureNum);
                convolvedFeatures[imageNum][featureNum] = convolvedFeature;
                //convolvedFeature.reshape(1,convolvedFeature.length);
                //convolvedImage = DoubleMatrix.concatHorizontally(convolvedImage, convolvedFeature);
            }
            //convolvedFeatures.putRow(imageNum, convolvedImage);
        }
        return convolvedFeatures;
    }


	private DoubleMatrix convolve() {
		System.out.println("Starting Convolution");
		int numFeatures = whitenedTheta.columns;
		this.resultRows = (imageRows-patchRows+1)/poolRows;
		this.resultCols = (imageCols-patchCols+1)/poolCols;
		this.imageSize = imageRows*imageCols;
		this.pooledFeatures = new DoubleMatrix(images.length, numFeatures * (resultRows * resultCols));
        //this.convolvedFeatures = new DoubleMatrix(images.rows, numFeatures * imageRows * imageCols);
		ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
		for(int imageNum = 0; imageNum < images.length; imageNum++) {
			Runnable worker = new ConvolutionThread(imageNum);
			executor.execute(worker);
		}
		executor.shutdown();
		while(!executor.isTerminated());
		return pooledFeatures;
        //return convolvedFeatures;
	}
	
	private class ConvolutionThread implements Runnable {
		private int imageNum;

		public ConvolutionThread(int imageNum) {
			this.imageNum = imageNum;
		}

		@Override
		public void run() {
			System.out.println("Image: " + imageNum);
			DoubleMatrix[] currentImage = images[imageNum];
            //DoubleMatrix convolvedImage = new DoubleMatrix(1,0);
			for (int featureNum = 0; featureNum < whitenedTheta.columns; featureNum++) {
				DoubleMatrix convolvedFeature = convFeature(currentImage, whitenedTheta, featureNum);
                //convolvedImage = DoubleMatrix.concatHorizontally(convolvedImage, convolvedFeature);
                //convolvedFeatures.putRow(imageNum, convolvedFeature);
				pool(convolvedFeature, imageNum, featureNum);
			}
            //convolvedFeatures.putRow(imageNum, convolvedImage);
		}

	}
	
	public DoubleMatrix convFeature(DoubleMatrix[] currentImage, DoubleMatrix features, int featureNum) {
		DoubleMatrix convolvedFeature = DoubleMatrix.zeros(imageRows-patchRows+1,imageCols - patchCols+1);
		for(int channel = 0; channel < currentImage.length; channel++) {
			DoubleMatrix feature = features.getRange(patchSize*channel, patchSize*channel+patchSize,featureNum, featureNum+1);
			feature.reshape(patchRows, patchCols);
            DoubleMatrix image = currentImage[channel];
			DoubleMatrix conv = Utils.conv2d(image, feature);
			convolvedFeature.addi(conv);
		}
		return Utils.sigmoid(convolvedFeature.add(features.get(featureNum)));
	}

	public void pool(DoubleMatrix convolvedFeature, int imageNum, int featureNum) {
		for(int poolRow = 0; poolRow < resultRows; poolRow++) {
			for(int poolCol = 0; poolCol < resultCols; poolCol++) {
				DoubleMatrix patch = convolvedFeature.getRange(poolRow*poolRows, poolRow*poolRows+poolRows, poolCol*poolCols, poolCol*poolCols+poolCols);
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
            whitenedBias = new DoubleMatrix(Integer.parseInt(line[0]), Integer.parseInt(line[1]));
            line = reader.readLine().split(",");
			for(int i = 0; i < whitenedBias.rows; i++) {
				for(int j = 0; j < whitenedBias.columns; j++) {
					whitenedBias.put(i, j, Double.parseDouble(line[i * whitenedBias.columns + j]));
				}
			}
			imageRows = Integer.parseInt(reader.readLine());
			imageCols = Integer.parseInt(reader.readLine());
			patchRows = Integer.parseInt(reader.readLine());
            patchCols = Integer.parseInt(reader.readLine());
			poolRows = Integer.parseInt(reader.readLine());
            poolCols = Integer.parseInt(reader.readLine());
			reader.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}

    @Override
    public DataContainer feedForward(DataContainer input) {
        return new DataContainer(oonv(input.getDataArray(), whitenedTheta));
    }

    @Override
<<<<<<< HEAD
    public DoubleMatrix backPropagation(DataContainer[] results, int layer, DoubleMatrix y, double momentum, double alpha) {
        System.out.println(results[layer-1].getDataArray().length+":"+results[layer-1].getDataArray()[0].length+":"+results[layer-1].getDataArray()[0][0].rows+":"+results[layer-1].getDataArray()[0][0].columns);
        System.out.println(y.rows+":"+ y.columns);
        DoubleMatrix delta = Utils.flatten(oonv(results[layer-1].getDataArray(), y));
        whitenedTheta.subi(delta);
        return delta;
=======
    public DoubleMatrix backPropagation(DoubleMatrix[] results, int layer, DoubleMatrix y, double momentum, double alpha) {
        DoubleMatrix delta = oonv(results[layer-1], y);
>>>>>>> parent of c90cee4... Full network backprop implemented (not tested)
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
			writer.write("\n"+imageRows+"\n"+imageCols+"\n"+patchRows+"\n"+patchCols+"\n"+poolRows+"\n"+poolCols);
			writer.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	
	public DataContainer compute(DataContainer input) {
		this.images = input.getDataArray();
		this.pooledFeatures = convolve();
		return new DataContainer(this.pooledFeatures);
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
	public DataContainer train(DataContainer input, DoubleMatrix output, int iterations) {
        //int inputSize = patchDim*patchDim*channels;
		//SparseAutoencoder ae = new SparseAutoencoder(inputSize, numPatches, inputSize, sparsityParam, lambda, beta, alpha);
        LinearDecoder ae = new LinearDecoder(patchRows, patchCols, channels, numPatches, sparsityParam, lambda, beta, alpha, Utils.PRELU, Utils.NONE);
		DataContainer patch = ImageLoader.sample(patchRows, patchCols, 100000, input, imageCols, imageRows, channels);
        DoubleMatrix patches = patch.getData();
        //try {
            if(whiten) {
                patches.divi(patches.max());
                DoubleMatrix meanPatch = patches.columnMeans();
                DoubleMatrix ZCAWhite = Utils.calculateZCAWhite(patches, meanPatch, 0.1);
                patches = Utils.ZCAWhiten(patches, meanPatch, ZCAWhite);
                patch.update(patches);
                ae.train(patch, patches, iterations);
                DoubleMatrix previousTheta = ae.getTheta();
                DoubleMatrix previousBias = ae.getBias();
                whitenedTheta = ZCAWhite.mmul(previousTheta);
                whitenedBias = previousBias.sub(meanPatch.mmul(whitenedTheta));
            }
            else {
                ae.train(patch, patches, iterations);
                this.whitenedTheta = ae.getTheta();
                this.whitenedBias = ae.getBias();
            }
        /*} catch (IOException e) {
            e.printStackTrace();
        }*/
		return compute(input);
		
	}
}
