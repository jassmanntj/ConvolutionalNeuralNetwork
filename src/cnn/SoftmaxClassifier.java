package cnn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import edu.stanford.nlp.optimization.DiffFunction;
import edu.stanford.nlp.optimization.QNMinimizer;

public class SoftmaxClassifier extends NeuralNetworkLayer implements DiffFunction{
	private static final boolean DEBUG = true;
	private int inputSize;
	private int outputSize;
	private int m;
    private int T = 1;
	private double lambda;
    private double alpha;
    private double cost;
	private DoubleMatrix theta;
	private DoubleMatrix input;
	private DoubleMatrix output;
    DoubleMatrix t1Velocity;
    private int activation = Utils.SIGMOID;
	
	public SoftmaxClassifier(double lambda, double alpha) {
		this.lambda = lambda;
        this.alpha = alpha;
        this.cost = 0;
        initializeParams();
	}
	
	public DoubleMatrix getTheta() {
		return theta;
	}
	
	private void initializeParams() {
		double r = Math.sqrt(6)/Math.sqrt(inputSize+1);
		theta = DoubleMatrix.rand(inputSize, outputSize).muli(2 * r).subi(r);
        this.t1Velocity = new DoubleMatrix(theta.rows, theta.columns);
	}
	
	public DoubleMatrix computeNumericalGradient(DoubleMatrix input, DoubleMatrix output) {
		double epsilon = 0.0001;
		DoubleMatrix numGrad = DoubleMatrix.zeros(theta.rows, theta.columns);
		for(int i = 0; i < theta.rows; i++) {
			for(int j = 0; j < theta.columns; j++) {
				DoubleMatrix thetaPlus = theta.dup();
				DoubleMatrix thetaMinus = theta.dup();
				thetaPlus.put(i,j,thetaPlus.get(i,j)+epsilon);
				thetaMinus.put(i,j,thetaMinus.get(i,j)-epsilon);
				CostResult costPlus = cost(input, output, thetaPlus);
				CostResult costMinus = cost(input, output, thetaMinus);
				numGrad.put(i,j,(costPlus.cost-costMinus.cost)/(2*epsilon));
			}
		}
		return numGrad;
	}
	
	public CostResult cost(DoubleMatrix input, DoubleMatrix output, DoubleMatrix theta) {
		m = input.rows;
		DoubleMatrix res = input.mmul(theta);
		DoubleMatrix p = res.subColumnVector(res.rowMaxs());
		MatrixFunctions.expi(p);
		p.diviColumnVector(p.rowSums());

		DoubleMatrix thetaGrad =input.transpose().mmul(p.sub(output)).div(m).add(theta.mul(lambda));
		DoubleMatrix delta = p.sub(output).mmul(theta.transpose());
        Utils.activationGradient(activation, input, delta);
		MatrixFunctions.logi(p);
		cost = -p.mul(output).sum()/m + theta.mul(theta).sum()*lambda/2;
		return new CostResult(cost, thetaGrad, null, delta);
	}

    public double getCost() {
        return cost;
    }

    public DoubleMatrix backPropagation(DoubleMatrix[] results, int layer, DoubleMatrix y, double momentum, double alpha) {
        CostResult res = cost(results[layer-1], y, theta);
        t1Velocity.muli(momentum).addi(res.thetaGrad.mul(alpha));
        theta.subi(t1Velocity);
        return res.delta;
    }

    public DoubleMatrix feedForward(DoubleMatrix input) {
        return input.mmul(theta);
    }

    public CostResult cost(DoubleMatrix input, DoubleMatrix output) {
        return cost(input, output, theta);
    }

    public void miniBatchGradientDescent(DoubleMatrix input, DoubleMatrix output, int iterations, final double momentum, int batchSize) {
        m = input.rows;
        System.out.println("INPUT Rows: "+input.rows+" Cols: "+input.columns);
        System.out.println("Starting gradient descent with " + iterations + " iterations.");
        System.out.println("-----");
        final DoubleMatrix t1Velocity = new DoubleMatrix(theta.rows, theta.columns);
        class StochasticThread implements Runnable{
            DoubleMatrix batch;
            DoubleMatrix outBatch;
            int i;
            public StochasticThread(DoubleMatrix batch, DoubleMatrix outBatch, int i) {
                this.batch = batch;
                this.outBatch = outBatch;
                this.i = i;
            }
            public void run() {
                double curAlpha = alpha/(1+(i/T));
                CostResult result = cost(batch, outBatch, theta);
                t1Velocity.muli(momentum).addi(result.thetaGrad.mul(curAlpha));
                theta.subi(t1Velocity);
            }
        }
        for(int i = 0; i < iterations; i++) {
            System.out.println(i);
            if(i%100==0) {
                CostResult result = cost(input, output, theta);
                System.out.println("Cost "+i+": " + result.cost);
            }
            ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
            for(int j = 0; j < (input.rows + batchSize-1)/batchSize; j++) {
                int size = Math.min(batchSize, input.rows-j*batchSize);
                DoubleMatrix batch = input.getRange(j*batchSize, j*batchSize+size, 0, input.columns);
                DoubleMatrix outBatch = output.getRange(j*batchSize, j*batchSize+size, 0, output.columns);
                Runnable worker = new StochasticThread(batch, outBatch, i);
                executor.execute(worker);
            }
            executor.shutdown();
            while(!executor.isTerminated());
        }
        CostResult res = cost(input, output, theta);
        System.out.println("Final Cost: " + res.cost);
    }

    public DoubleMatrix getA() {
        return null;
    }

	public void gradientDescent(DoubleMatrix input, DoubleMatrix output, int iterations, double alpha) {
		m = input.rows;
		System.out.println("INPUT Rows: "+input.rows+" Cols: "+input.columns);
		initializeParams();
		System.out.println("Starting gradient descent with " + iterations + " iterations.");
		System.out.println("-----");
		for(int i = 0; i < iterations; i++) {
			CostResult result = cost(input, output, theta);
			if(DEBUG) {
				DoubleMatrix numGrad = computeNumericalGradient(input, output);
				DoubleMatrix gradMin = numGrad.dup();
				DoubleMatrix gradAdd = numGrad.dup();
				gradMin.subi(result.thetaGrad);
				gradAdd.addi(result.thetaGrad);
				System.out.println("Diff: "+gradMin.norm2()/gradAdd.norm2());
			}
			System.out.println("Interation " + i + " Cost: " + result.cost);
			theta.subi(result.thetaGrad.mul(alpha));
		}
	}
	
	public void lbfgsTrain(DoubleMatrix input, DoubleMatrix output, int iterations) {
		this.input = input;
		this.output = output;
		m = input.rows;
		System.out.println("INPUT Rows: "+input.rows+" Cols: "+input.columns);
		initializeParams();
		System.out.println("Starting lbfgs.");
		System.out.println("-----");
		QNMinimizer qn = new QNMinimizer(25, true);
		theta.data = qn.minimize(this, 1e-9, theta.data, iterations);
	}
	
	public void writeTheta(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			writer.write(theta.rows+","+theta.columns+"\n");
			for(int i = 0; i < theta.length; i++){
				if( i < theta.length-1)
					writer.write(theta.data[i]+",");
				else writer.write(""+theta.data[i]);
			}
			writer.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public void writeLayer(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			writer.write(theta.rows+","+theta.columns+"\n");
			for(int i = 0; i < theta.rows; i++){
				for(int j = 0; j < theta.columns; j++) {
					writer.write(theta.get(i,j)+",");
				}
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
			String[] thetaSize = reader.readLine().split(",");
			theta = new DoubleMatrix(Integer.parseInt(thetaSize[0]),Integer.parseInt(thetaSize[1]));
			String[] data = reader.readLine().split(",");
			assert data.length == theta.data.length;
			for(int i = 0; i < theta.rows; i++) {
				for(int j = 0; j < theta.columns; j++) {
					theta.put(i, j, Double.parseDouble(data[i * theta.columns + j]));
				}
			}
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
			String[] thetaSize = reader.readLine().split(",");
			theta = new DoubleMatrix(Integer.parseInt(thetaSize[0]),Integer.parseInt(thetaSize[1]));
			String[] data = reader.readLine().split(",");
			assert data.length == theta.data.length;
			for(int i = 0; i < theta.data.length; i++) {
				theta.data[i] = Double.parseDouble(data[i]);
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		return compute(input);
	}
	
	public void loadTheta(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] thetaSize = reader.readLine().split(",");
			theta = new DoubleMatrix(Integer.parseInt(thetaSize[0]),Integer.parseInt(thetaSize[1]));
			String[] data = reader.readLine().split(",");
			assert data.length == theta.data.length;
			for(int i = 0; i < theta.data.length; i++) {
				theta.data[i] = Double.parseDouble(data[i]);
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public int[] computeResults(DoubleMatrix input) {
		DoubleMatrix result = input.mmul(theta);
		int[] results = new int[result.rows];
		for(int i = 0; i < result.rows; i++) {
			double currentMax = 0;
			for(int j = 0; j < result.columns; j++) {
				if(result.get(i,j) > currentMax) {
					currentMax = result.get(i,j);
					results[i] = j;
				}
			}
		}
		return results;
	}
	
	@Override
	public int domainDimension() {
		return theta.length;
	}

	@Override
	public double valueAt(double[] arg0) {
		theta.data = arg0;
		CostResult res = cost(input, output, theta);
		return res.cost;
	}

	@Override
	public double[] derivativeAt(double[] arg0) {
		theta.data = arg0;
		CostResult res = cost(input, output, theta);
		return res.thetaGrad.data;
	}

	@Override
	public DoubleMatrix train(DoubleMatrix input, DoubleMatrix output, int iterations) {
		inputSize = input.columns;
		outputSize = output.columns;
		initializeParams();
        //miniBatchGradientDescent(input, output, iterations, 0.9, 128);
		lbfgsTrain(input, output, iterations);
		return compute(input);
		
	}

	@Override
	public DoubleMatrix compute(DoubleMatrix input) {
		return input.mmul(theta);
	}

	@Override
	public DoubleMatrix getBias() {
		return null;
	}
}
