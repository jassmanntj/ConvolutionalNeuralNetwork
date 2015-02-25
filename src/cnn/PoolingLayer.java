package cnn;

import org.jblas.DoubleMatrix;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by jassmanntj on 2/23/2015.
 */
public class PoolingLayer extends NeuralNetworkLayer {
    private int poolDimX;
    private int poolDimY;
    private int resultRows;
    private int resultCols;
    private int features;


    public PoolingLayer(int poolDim, int features) {
        this.poolDimX = poolDim;
        this.poolDimY = poolDim;
        this.features = features;
    }

    public PoolingLayer(int poolDimX, int poolDimY, int features) {
        this.poolDimX = poolDimX;
        this.poolDimY = poolDimY;
        this.features = features;
    }

    public DoubleMatrix compute(DoubleMatrix input) {
        this.resultRows = input.rows/poolDimX;
        this.resultCols = input.columns/poolDimY;
        final DoubleMatrix pooledFeatures = new DoubleMatrix(resultRows, resultCols);
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);

        class PoolingThread implements Runnable {
            private int imageNum;
            private int featureNum;
            private DoubleMatrix feature;

            public PoolingThread(int imageNum, int featureNum, DoubleMatrix feature) {
                this.imageNum = imageNum;
                this.featureNum = featureNum;
                this.feature = feature;
            }

            @Override
            public void run() {
                pool(feature, imageNum, featureNum, pooledFeatures);
            }
        }

        for(int imageNum = 0; imageNum < input.rows; imageNum++) {
            for(int featureNum = 0; featureNum < features; featureNum++) {
                DoubleMatrix feature = input.getRange(imageNum,imageNum,1,2);
                Runnable worker = new PoolingThread(imageNum, featureNum, feature);
                executor.execute(worker);
            }
        }
        return pooledFeatures;
    }

    public void pool(DoubleMatrix convolvedFeature, int imageNum, int featureNum, DoubleMatrix pooledFeatures) {
        for(int poolRow = 0; poolRow < resultRows; poolRow++) {
            for(int poolCol = 0; poolCol < resultCols; poolCol++) {
                DoubleMatrix patch = convolvedFeature.getRange(poolRow*poolDimX, poolRow*poolDimX+poolDimX, poolCol*poolDimY, poolCol*poolDimY+poolDimY);
                pooledFeatures.put(imageNum, featureNum*resultRows*resultCols+poolRow*resultCols+poolCol, patch.mean());
            }
        }
    }

    public DoubleMatrix getTheta() {
        return null;
    }
    public DoubleMatrix getBias() {
        return null;
    }
    public void writeTheta(String filename) {
    }
    public DoubleMatrix loadTheta(String filename, DoubleMatrix input) {
        return compute(input);
    }
    public DoubleMatrix train(DoubleMatrix input, DoubleMatrix output, int iterations) throws IOException {
        return compute(input);
    }
    public void writeLayer(String filename) {
    }
    public void loadLayer(String filename) {
    }
}
