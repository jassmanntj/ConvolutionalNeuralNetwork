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
    private int stepX;
    private int stepY;
    private int resultRows;
    private int resultCols;

    public DoubleMatrix getA() {
        return null;
    }


    public PoolingLayer(int poolDim, int features) {
        this.poolDimX = poolDim;
        this.poolDimY = poolDim;
        this.stepX = poolDim;
        this.stepY = poolDim;
    }

    public PoolingLayer(int poolDimX, int poolDimY, int features) {
        this.poolDimX = poolDimX;
        this.poolDimY = poolDimY;
        this.stepX = poolDimX;
        this.stepY = poolDimY;
    }

    public DataContainer feedForward(DataContainer input) {
        return compute(input);
    }

    @Override
    public DoubleMatrix backPropagation(DataContainer[] results, int layer, DoubleMatrix y, double momentum, double alpha) {
        DoubleMatrix delta = expand(y);
        return delta;
    }

    public DataContainer compute(DataContainer in) {
        DoubleMatrix[][] input = in.getDataArray();
        this.resultRows = input[0][0].rows/stepX;
        this.resultCols = input[0][0].columns/stepY;
        DoubleMatrix pooledFeatures[][] = new DoubleMatrix[input.length][input[0].length];

        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            for(int featureNum = 0; featureNum < input[imageNum].length; featureNum++) {
                pooledFeatures[imageNum][featureNum] = pool(input[imageNum][featureNum]);
            }
        }
        return new DataContainer(pooledFeatures);
    }

    public DoubleMatrix pool(DoubleMatrix convolvedFeature) {
        DoubleMatrix pooledFeature = new DoubleMatrix(resultRows, resultCols);
        for(int poolRow = 0; poolRow < resultRows; poolRow++) {
            for(int poolCol = 0; poolCol < resultCols; poolCol++) {
                DoubleMatrix patch = convolvedFeature.getRange(poolRow*stepX, poolRow*stepX+poolDimX, poolCol*stepY, poolCol*stepY+poolDimY);
                pooledFeature.put(poolRow, poolCol, patch.mean());
            }
        }
        return pooledFeature;
    }

    private DoubleMatrix expand(DoubleMatrix in) {
        System.out.println(in.rows+"::"+in.columns);
        DoubleMatrix expandedMatrix = new DoubleMatrix(in.rows*stepX, in.columns*stepY);
        double scale = (poolDimX * poolDimX * poolDimY * poolDimY / (stepX * stepY));
        for(int i = 0; i < in.rows; i++) {
            for(int j = 0; j < in.columns; j++) {
                double value = in.get(i,j)/scale;
                for(int k = 0; k < poolDimY; k++) {
                    for(int l = 0; l < poolDimX; l++) {
                        expandedMatrix.put(i*stepY+k, j*stepX+l, expandedMatrix.get(i*stepY+k, j*stepX+l)+value);
                    }
                }
            }
        }
        return expandedMatrix;
    }

    public DoubleMatrix getTheta() {
        return null;
    }
    public DoubleMatrix getBias() {
        return null;
    }
    public void writeTheta(String filename) {
    }
    public DataContainer train(DataContainer input, DoubleMatrix output, int iterations) throws IOException {
        return compute(input);
    }
    public void writeLayer(String filename) {
    }
    public void loadLayer(String filename) {
    }
}
