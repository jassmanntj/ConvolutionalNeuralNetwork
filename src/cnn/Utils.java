package cnn;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.Singular;
import org.jtransforms.fft.DoubleFFT_2D;

public class Utils {
    public static final int NUMTHREADS = 8;
    public static final int NONE = 0;
    public static final int SIGMOID = 1;
    public static final int PRELU = 2;
    public static final int RELU = 3;
	public static DoubleMatrix calculateZCAWhite(DoubleMatrix input, DoubleMatrix meanPatch, double epsilon) {
<<<<<<< HEAD
        DoubleMatrix sigma = input.subRowVector(meanPatch);
        sigma = sigma.transpose().mmul(sigma);
=======
		DoubleMatrix sigma = input.subRowVector(meanPatch);
		sigma = sigma.transpose().mmul(sigma);
>>>>>>> parent of e6651bd... Working version
		sigma.divi(input.rows);
		DoubleMatrix[] svd = Singular.fullSVD(sigma);
		DoubleMatrix ZCAWhite = svd[1];
		ZCAWhite.addi(epsilon);
		ZCAWhite.rdivi(1);
		ZCAWhite = DoubleMatrix.diag(ZCAWhite);
		ZCAWhite = svd[0].mmul(ZCAWhite).mmul(svd[0].transpose());
		return ZCAWhite;
	}
	
	public static DoubleMatrix ZCAWhiten(DoubleMatrix input, DoubleMatrix meanPatch, DoubleMatrix ZCAWhite) {
		input.subiRowVector(meanPatch);
		input = input.mmul(ZCAWhite); 
		return input;
	}
	
	public static DoubleMatrix conv2d(DoubleMatrix input, DoubleMatrix kernel) {
		int inputRows = input.rows;
		int inputCols = input.columns;
		int kernelRows = kernel.rows;
		int kernelCols = kernel.columns;
		int totalRows = inputRows + kernelRows - 1;
		int totalCols = inputCols + kernelCols - 1;
		reverseMatrix(kernel);
		input = DoubleMatrix.concatHorizontally(input, DoubleMatrix.zeros(input.rows, kernel.columns-1));
		input = DoubleMatrix.concatVertically(input, DoubleMatrix.zeros(kernel.rows-1, input.columns));
		kernel = DoubleMatrix.concatHorizontally(kernel, DoubleMatrix.zeros(kernel.rows, input.columns-kernel.columns));
		kernel = DoubleMatrix.concatVertically(kernel, DoubleMatrix.zeros(input.rows-kernel.rows,kernel.columns));
		ComplexDoubleMatrix inputDFT = new ComplexDoubleMatrix(input);
		ComplexDoubleMatrix kernelDFT = new ComplexDoubleMatrix(kernel);
		DoubleFFT_2D t = new DoubleFFT_2D(inputDFT.rows, inputDFT.columns);
		t.complexForward(inputDFT.data);
		t.complexForward(kernelDFT.data);
		kernelDFT.muli(inputDFT);
		t.complexInverse(kernelDFT.data, true);
		int rowSize = inputRows - kernelRows + 1;
		int colSize = inputCols - kernelCols + 1;
		DoubleMatrix result = kernelDFT.getReal();
		int startRows = (totalRows-rowSize)/2;
		int startCols = (totalCols-colSize)/2;
		result = result.getRange(startRows, startRows+rowSize,startCols, startCols+colSize);
		return result;
	}
	
	private static DoubleMatrix reverseMatrix(DoubleMatrix mat) {
		for(int i = 0; i < mat.rows/2; i++) {
			mat.swapRows(i, mat.rows-i-1);
		}
		for(int i = 0; i < mat.columns/2; i++) {
			mat.swapColumns(i, mat.columns-i-1);
		}
		return mat;
	}
	
	public static void visualizeColor(int width, int height, int images, DoubleMatrix img, String filename) throws IOException {
		BufferedImage image = new BufferedImage(width*images+images*2+2, height*images+images*2+2, BufferedImage.TYPE_INT_RGB);
		DoubleMatrix tht1 = img.dup();
		tht1.subi(tht1.min());
		tht1.divi(tht1.max());
		tht1.muli(255);
		for(int k = 0; k < images; k++) {
			for(int l = 0; l < images; l++) {
				if(k*images+l < tht1.rows) { 
					DoubleMatrix row = tht1.getRow(k*images+l);
					int channelSize = row.length/3;
					double[] r = new double[channelSize];
					double[] g = new double[channelSize];
					double[] b = new double[channelSize];
					System.arraycopy(row.data, 0, r, 0, channelSize);
					System.arraycopy(row.data, channelSize, g, 0, channelSize);
					System.arraycopy(row.data, 2*channelSize, b, 0, channelSize);
					for(int i = 0; i < height; i++) {
						for(int j = 0; j < width; j++) {
							int col = ((int)r[i*width+j] << 16) | ((int)g[i*width+j] << 8) | (int)b[i*width+j];
							image.setRGB(l*(width+2)+2+j, k*(height+2)+2+i, col);
						}
					}
				}
			}
		}
		File imageFile = new File(filename);
		ImageIO.write(image, "png", imageFile);
	}
	
	public static void visualize(int size, int images, DoubleMatrix input, String filename) throws IOException {
		BufferedImage image = new BufferedImage((size+2)*images+2, (size+2)*images+2, BufferedImage.TYPE_INT_RGB);
		DoubleMatrix tht1 = input.dup();
		tht1.subi(tht1.min());
		tht1.divi(tht1.max());
		tht1.muli(255);
		System.out.println(tht1.getRow(0));
		for(int k = 0; k < images; k++) {
			for(int l = 0; l < images; l++) {
				for(int i = 0; i < size; i++) {
					for(int j = 0; j < size; j++) {
						int imageNo = (int)(Math.random() * 9999);
						int val = (int)tht1.get(imageNo,i*size+j);
						int col = (val << 16) | (val << 8) | val;
						image.setRGB(l*(2+size)+2+j, k*(2+size)+2+i, col);
					}
				}
			}
		}
		File imageFile = new File(filename);
		ImageIO.write(image, "png", imageFile);
	}

    public static DoubleMatrix activationFunction(int type, DoubleMatrix z, double a) {
        switch(type) {
            case SIGMOID:
                return sigmoid(z);
            case PRELU:
                return prelu(z, a);
            case RELU:
                return relu(z);
            case NONE:
                return z;
            default:
                return sigmoid(z);
        }
    }

    public static DoubleMatrix activationGradient(int type, DoubleMatrix z, double a, DoubleMatrix delta) {
        switch(type) {
            case SIGMOID:
                return delta.muli(sigmoidGradient(z));
            case PRELU:
                return delta.muli(preluGradient(z, a));
            case RELU:
                return delta.muli(reluGradient(z));
            case NONE:
                return delta;
            default:
                return delta.muli(sigmoidGradient(sigmoid(z)));
        }
    }

    public static double aGrad(int type, DoubleMatrix result, double a) {
        return aGrad(type, result, null, a);
    }

    public static double aGrad(int type, DoubleMatrix result, DoubleMatrix delta, double a) {
        switch(type) {
            case PRELU:
                if(delta != null) return result.le(0).mul(result).mul(delta).mean();
                else return result.le(0).mul(result).mean()/a;
            case SIGMOID: return 0;
            case RELU: return 0;
            case NONE: return 0;
            default: return 0;
        }
    }

    public static DoubleMatrix flatten(DoubleMatrix[][] z) {
        DoubleMatrix images = null;
        for(int i = 0; i < z.length; i++) {
            DoubleMatrix image = null;
            for(int j = 0; j < z[i].length; j++) {
                for(int k = 0; k < z[i][j].rows; k++) {
                    if(image == null) image = z[i][j].getRow(k);
                    else image = DoubleMatrix.concatHorizontally(image, z[i][j].getRow(k));
                }
            }
            if(images == null) images = image;
            else images = DoubleMatrix.concatVertically(images, image);
        }
        return images;
    }

    public static DoubleMatrix activationFunction(int type, DoubleMatrix z) {
        return activationFunction(type, z, 0.25);
    }

    public static DoubleMatrix activationGradient(int type, DoubleMatrix z, DoubleMatrix delta) {
        return activationGradient(type, z, 0.25, delta);
    }

    public static DoubleMatrix prelu(DoubleMatrix z, double a) {
       //return z.le(0).mul(a-1).add(1).mul(z);
        DoubleMatrix res = z.dup();
        for(int i = 0; i < res.rows; i++) {
            for(int j = 0; j < res.columns; j++) {
                double k = res.get(i,j);
                if(k < 0) res.put(i,j, a*k);
            }
        }
        return res;
    }

    public static DoubleMatrix relu(DoubleMatrix z) {
        return prelu(z, 0);
    }

    public static DoubleMatrix preluGradient(DoubleMatrix z, double a) {
        //return z.le(0).mul(a-1).add(1);
        DoubleMatrix res = new DoubleMatrix(z.rows, z.columns);
        for(int i = 0; i < z.rows; i++) {
            for(int j = 0; j < z.columns; j++) {
                double k = z.get(i,j);
                if(k > 0) res.put(i,j,1);
                else res.put(i,j,a);
            }
        }
        return res;
    }

    public static DoubleMatrix reluGradient(DoubleMatrix z) {
        return z.gt(0);
    }
	
	public static DoubleMatrix sigmoid(DoubleMatrix z) {
		return MatrixFunctions.exp(z.neg()).add(1).rdiv(1);
	}
	
	public static DoubleMatrix sigmoidGradient(DoubleMatrix a) {
        return a.neg().add(1).mul(a);
	}
	
	public static int[][] computeResults(DoubleMatrix[][] result) {
		int[][] results = new int[result.length][3];
		for(int i = 0; i < result.length; i++) {
			double current1 = 0;
			double current2 = 0;
			double current3 = 0;
			for(int j = 0; j < result[i][0].length; j++) {
				if(result[i][0].get(j) > current1) {
					current3 = current2;
					current2 = current1;
					current1 = result[i][0].get(j);
					results[i][2] = results[i][1];
					results[i][1] = results[i][0];
					results[i][0] = j;
				}
				else if(result[i][0].get(j) > current2) {
					current3 = current2;
					current2 = result[i][0].get(j);
					results[i][2] = results[i][1];
					results[i][1] = j;
				}
				else if(result[i][0].get(j) > current3) {
					current3 = result[i][0].get(j);
					results[i][2] = j;
				}
				
			}
		}
		return results;
	}
	
}
