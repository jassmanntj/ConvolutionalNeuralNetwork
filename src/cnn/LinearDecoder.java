package cnn;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.imageio.ImageIO;

import org.jblas.DoubleMatrix;

import edu.stanford.nlp.optimization.DiffFunction;
import edu.stanford.nlp.optimization.QNMinimizer;

public class LinearDecoder extends NeuralNetworkLayer implements DiffFunction{
	private static final boolean DEBUG = false;
    private int T = 1;
	private int inputSize;
	private int hiddenSize;
	private double rho;
	private double lambda;
	private double beta;
	private double alpha;
    private double aLambda = 0;
	private DoubleMatrix theta1;
	private DoubleMatrix theta2;
	private DoubleMatrix bias1;
	private DoubleMatrix bias2;
	private DoubleMatrix input;
    private DoubleMatrix a1;
    private DoubleMatrix a2;
    private DoubleMatrix tVelocity;
    private DoubleMatrix bVelocity;
    private DoubleMatrix aVelocity;
	private CostResult[] currentCost;
	private int patchRows;
    private int patchColumns;
    private int patchStepX;
    private int patchStepY;
    private int layer1;
    private int layer2;
    private double epsilon;

	public LinearDecoder(int patchRows, int patchColumns, int channels, int hiddenSize, double sparsityParam, double lambda, double beta, double alpha, int layer1Func, int layer2Func) {
		this.inputSize = patchRows*patchColumns*channels;
		this.patchRows = patchRows;
        this.patchColumns = patchColumns;
        this.patchStepX = patchColumns;
        this.patchStepY = patchRows;
		this.hiddenSize = hiddenSize;
        this.layer1 = layer1Func;
        this.layer2 = layer2Func;
		this.lambda = lambda;
		this.beta = beta;
		this.alpha = alpha;
		this.rho = sparsityParam;
		initializeParams();
	}
	
	private void initializeParams() {
		double r = Math.sqrt(6)/Math.sqrt(hiddenSize+inputSize+1);
		theta1 = DoubleMatrix.rand(inputSize, hiddenSize).muli(2*r).subi(r);
        tVelocity = DoubleMatrix.zeros(inputSize, hiddenSize);
		theta2 = DoubleMatrix.rand(hiddenSize, inputSize).muli(2*r).subi(r);
		bias1 = DoubleMatrix.zeros(1, hiddenSize);
        bVelocity = DoubleMatrix.zeros(1, hiddenSize);
		bias2 = DoubleMatrix.zeros(1, inputSize);
        a1 =  new DoubleMatrix(1,1);
        aVelocity = DoubleMatrix.zeros(1,1);
        a1.put(0,0,0.25);
        a2 = new DoubleMatrix(1, 1);
        a2.put(0,0,0.25);
	}

    public DoubleMatrix getA() {
        return a1;
    }

	public DoubleMatrix getTheta() {
		return theta1;
	}
	
	public DoubleMatrix getBias() {
		return bias1;
	}
	
	private DoubleMatrix computeNumericalGradient(DataContainer input, DoubleMatrix output) {
		double epsilon = .0001;
		DoubleMatrix compiledMatrix = DoubleMatrix.zeros(1, theta1.length+theta2.length+bias1.length+bias2.length);
		for(int i = 0; i < compiledMatrix.length; i++) {
			if(i%1000==0)
				System.out.println(i+"/"+(compiledMatrix.length));
			if(i < theta1.length) {
				int j = i/theta1.columns;
				int k = i%theta1.columns;
				DoubleMatrix thetaPlus = theta1.dup();
				DoubleMatrix thetaMinus = theta1.dup();
				thetaPlus.put(j,k,thetaPlus.get(j,k)+epsilon);
				thetaMinus.put(j,k,thetaMinus.get(j,k)-epsilon);
				CostResult[] costPlus = cost(input, thetaPlus, theta2, bias1, bias2);
				CostResult[] costMinus = cost(input, thetaMinus, theta2, bias1, bias2);
				compiledMatrix.put(0,i,(costPlus[1].cost-costMinus[1].cost)/(2*epsilon));
			}
			else if(i < theta1.length + theta2.length) {
				int j = (i-theta1.length)/theta2.columns;
				int k = (i-theta1.length)%theta2.columns;
				DoubleMatrix thetaPlus = theta2.dup();
				DoubleMatrix thetaMinus = theta2.dup();
				thetaPlus.put(j,k,thetaPlus.get(j,k)+epsilon);
				thetaMinus.put(j,k,thetaMinus.get(j,k)-epsilon);
				CostResult[] costPlus = cost(input, theta1, thetaPlus, bias1, bias2);
				CostResult[] costMinus = cost(input, theta1, thetaMinus, bias1, bias2);
				compiledMatrix.put(0,i,(costPlus[1].cost-costMinus[1].cost)/(2*epsilon));
			}
			else if(i < theta1.length + theta2.length + bias1.length) {
				int j = (i-theta1.length-theta2.length)/bias1.columns;
				int k = (i-theta1.length-theta2.length)%bias1.columns;
				DoubleMatrix biasPlus = bias1.dup();
				DoubleMatrix biasMinus = bias1.dup();
				biasPlus.put(j,k,biasPlus.get(j,k)+epsilon);
				biasMinus.put(j,k,biasMinus.get(j,k)-epsilon);
				CostResult[] costPlus = cost(input, theta1, theta2, biasPlus, bias2);
				CostResult[] costMinus = cost(input, theta1, theta2, biasMinus, bias2);
				compiledMatrix.put(0,i,(costPlus[1].cost-costMinus[1].cost)/(2*epsilon));
			}
			else {
				int j = (i-theta1.length-theta2.length - bias1.length)/bias2.columns;
				int k = (i-theta1.length-theta2.length - bias1.length)%bias2.columns;
				DoubleMatrix biasPlus = bias2.dup();
				DoubleMatrix biasMinus = bias2.dup();
				biasPlus.put(j,k,biasPlus.get(j,k)+epsilon);
				biasMinus.put(j,k,biasMinus.get(j,k)-epsilon);
				CostResult[] costPlus = cost(input, theta1, theta2, bias1, biasPlus);
				CostResult[] costMinus = cost(input, theta1, theta2, bias1, biasMinus);
				compiledMatrix.put(0,i,(costPlus[1].cost-costMinus[1].cost)/(2*epsilon));
			}
		}
		return compiledMatrix;
	}
	
	protected CostResult[] cost(DataContainer in, DoubleMatrix theta1, DoubleMatrix theta2, DoubleMatrix bias1, DoubleMatrix bias2) {
		DoubleMatrix input = in.getData();
        DoubleMatrix[] result = feedForward(input, theta1, theta2, bias1, bias2);
		DoubleMatrix[] thetaGrad = new DoubleMatrix[2];
		thetaGrad[0] = DoubleMatrix.zeros(theta1.rows, theta1.columns);
		thetaGrad[1] = DoubleMatrix.zeros(theta2.rows, theta2.columns);
		// squared error
		DoubleMatrix cost = result[3].sub(input);
		cost.muli(cost);
		double squaredErr = cost.sum() / (2 * input.rows);
        //System.out.println("----SQERR: "+squaredErr);
		//sparsity
        double sparsity = 0;
        DoubleMatrix means = null;
        if(layer1 == Utils.SIGMOID) {
            double klsum = 0;
            means = result[1].columnMeans();
            for (double rhohat : means.data) {
                klsum += rho * Math.log(rho / rhohat) + (1 - rho) * Math.log((1 - rho) / (1 - rhohat));
            }
            sparsity = klsum * beta;
        }
        //System.out.println("----SPARS: "+sparsity);
		//weightDecay
		double weightDecay = theta1.mul(theta1).sum();
		weightDecay += theta2.mul(theta2).sum();
		weightDecay *= lambda/2;
        weightDecay += aLambda/2 * (a2.get(0)*a2.get(0)+a1.get(0)*a1.get(0));
        //System.out.println("----WDEC: "+weightDecay);
		double costSum = squaredErr + weightDecay + sparsity;
		//delta3
		DoubleMatrix delta3 = result[3].sub(input);
        Utils.activationGradient(layer2, result[2], a2.get(0,0), delta3);
		//sparsity term
		//
		 DoubleMatrix betaTerm = DoubleMatrix.zeros(1,result[1].columns);
        if(layer1 == Utils.SIGMOID) {
            int i = 0;
            for (double rhohat : means.data) {
                double bterm = beta * (-rho / rhohat + (1 - rho) / (1 - rhohat));
                betaTerm.put(0, i++, bterm);
            }
        }
		//delta2
		DoubleMatrix delta2 = delta3.mmul(theta2.transpose());
        if(layer1 == Utils.SIGMOID) delta2.addiRowVector(betaTerm);
        Utils.activationGradient(layer1, result[0], a1.get(0,0), delta2);

		//W2grad
		thetaGrad[1] = result[1].transpose().mmul(delta3);
		thetaGrad[1].divi(input.rows);
		thetaGrad[1].addi(theta2.mul(lambda));
		//W1grad
		thetaGrad[0] = input.transpose().mmul(delta2);
		thetaGrad[0].divi(input.rows);
		thetaGrad[0].addi(theta1.mul(lambda));
		//b2grad
		DoubleMatrix[] biasGrad = new DoubleMatrix[2];
		biasGrad[1] = delta3.columnMeans();
		//b1grad
		biasGrad[0] = delta2.columnMeans();

        //agrad
        double a2grad = Utils.aGrad(layer2, result[2], a2.get(0));
        double a1grad = Utils.aGrad(layer1, result[0], delta2, a1.get(0));

		CostResult[] results = new CostResult[2];
		results[0] = new CostResult(0, 0, thetaGrad[0], biasGrad[0], delta2, a1grad);
		results[1] = new CostResult(costSum, squaredErr+weightDecay, thetaGrad[1], biasGrad[1], delta3, a2grad);
		return results;
	}

    public DoubleMatrix backPropagation(DataContainer[] result, int layer, DoubleMatrix y, double momentum, double alpha) {
        DoubleMatrix delta = y.mmul(theta1.transpose());
        DoubleMatrix results = result[layer].getData();
        if(layer1 == Utils.SIGMOID) {
            DoubleMatrix betaTerm = DoubleMatrix.zeros(1,results.columns);
            DoubleMatrix means = Utils.activationFunction(layer1, results, a1.get(0)).columnMeans();
            int i = 0;
            for (double rhohat : means.data) {
                double bterm = beta * (-rho / rhohat + (1 - rho) / (1 - rhohat));
                betaTerm.put(0, i++, bterm);
            }
            delta.addiRowVector(betaTerm);
        }
        //delta2
        DoubleMatrix prevResult = result[layer-1].getData();
        Utils.activationGradient(layer1, prevResult, a1.get(0), delta);
        DoubleMatrix thetaGrad = prevResult.transpose().mmul(y);
        thetaGrad.divi(prevResult.rows);
        thetaGrad.addi(theta1.mul(lambda));
        DoubleMatrix biasGrad = y.columnMeans();
        double aGrad = Utils.aGrad(layer1, prevResult, delta, a1.get(0));
        epsilon = alpha /100;
        tVelocity.muli(momentum).addi(thetaGrad.mul(alpha));
        bVelocity.muli(momentum).addi(biasGrad.mul(alpha));
        aVelocity.muli(momentum).addi(aGrad * epsilon);

        theta1.subi(tVelocity);
        bias1.subi(bVelocity);
        a1.subi(aVelocity);

        return delta;

    }
	
	
	public CostResult stackedCost(DoubleMatrix input, DoubleMatrix delta3, double lastAGrad) {
		//sparsity term
		//DoubleMatrix betaTerm = means.rdiv(-rho).add(means.rsub(1).rdiv(1-rho)).mul(beta);
		
		//delta2
		DoubleMatrix delta2 = delta3.mmul(theta1.transpose());

        Utils.activationGradient(layer1, input, delta2);
		
		//W1grad
		DoubleMatrix thetaGrad = input.transpose().mmul(delta3);
		thetaGrad.divi(input.rows);
		
		//b1grad
		DoubleMatrix biasGrad = delta3.columnMeans();

        double aGrad = Utils.aGrad(layer1, input.mmul(theta1), a1.get(0)) * lastAGrad;
		
		return new CostResult(0, 0, thetaGrad, biasGrad, delta2, aGrad);
	}
	
	public void lbfgsTrain(DoubleMatrix input, int iterations) {
		this.input = input;
		System.out.println("INPUT Rows: "+input.rows+" Cols: "+input.columns);
		System.out.println("Starting lbfgs.");
		System.out.println("-----");
		QNMinimizer qn = new QNMinimizer(15, true);
		double[] initial = new double[theta1.length+theta2.length+bias1.length+bias2.length+2];
		System.arraycopy(theta1.data, 0, initial, 0, theta1.data.length);
		System.arraycopy(theta2.data, 0, initial, theta1.data.length, theta2.data.length);
		System.arraycopy(bias1.data, 0, initial, theta1.data.length+theta2.data.length, bias1.data.length);
		System.arraycopy(bias2.data, 0, initial, theta1.data.length+theta2.data.length+bias1.data.length, bias2.data.length);
        initial[initial.length-1] = a2.get(0);
        initial[initial.length-2] = a1.get(0);
		initial = qn.minimize(this, 1e-5, initial, iterations);
		System.arraycopy(initial, 0, theta1.data, 0, theta1.data.length);
		System.arraycopy(initial, theta1.data.length, theta2.data, 0, theta2.data.length);
		System.arraycopy(initial, theta1.data.length+theta2.data.length, bias1.data, 0, bias1.data.length);
		System.arraycopy(initial, theta1.data.length+theta2.data.length+bias1.data.length, bias2.data, 0, bias2.data.length);
        a2.put(0,initial[initial.length-1]);
        a1.put(0,initial[initial.length-2]);
	}


    public void miniBatchGradientDescent(DataContainer in, int iterations, final double mom, int batchSize) {
        DoubleMatrix input = in.getData();
        System.out.println("INPUT Rows: "+input.rows+" Cols: "+input.columns);
        System.out.println("Starting gradient descent with " + iterations + " iterations.");
        System.out.println("-----");
        final DoubleMatrix t1Velocity = DoubleMatrix.zeros(theta1.rows, theta1.columns);
        final DoubleMatrix t2Velocity = DoubleMatrix.zeros(theta2.rows, theta2.columns);
        final DoubleMatrix b1Velocity = DoubleMatrix.zeros(bias1.rows, bias1.columns);
        final DoubleMatrix b2Velocity = DoubleMatrix.zeros(bias2.rows, bias2.columns);
        final DoubleMatrix a1Velocity = DoubleMatrix.zeros(1);
        final DoubleMatrix a2Velocity = DoubleMatrix.zeros(1);
        class StochasticThread implements Runnable{
            DoubleMatrix batch;
            int i;
            int inRows;
            int inCols;
            double momentum;
            public StochasticThread(DoubleMatrix batch, int i, int inRows, int inCols, double momentum) {
                this.batch = batch;
                this.i = i;
                this.inRows = inRows;
                this.inCols = inCols;
                this.momentum = momentum;
            }
            public void run() {
                CostResult[] result = cost(new DataContainer(batch), theta1, theta2, bias1, bias2);
                t1Velocity.muli(momentum).addi(result[0].thetaGrad.mul(alpha));
                t2Velocity.muli(momentum).addi(result[1].thetaGrad.mul(alpha));
                b1Velocity.muli(momentum).addi(result[0].biasGrad.mul(alpha));
                b2Velocity.muli(momentum).addi(result[1].biasGrad.mul(alpha));
                a1Velocity.muli(momentum).addi(result[0].getAGrad() * alpha);
                a2Velocity.muli(momentum).addi(result[1].getAGrad() * alpha);

                theta1.subi(t1Velocity);
                theta2.subi(t2Velocity);
                bias1.subi(b1Velocity);
                bias2.subi(b2Velocity);
                a1.subi(a1Velocity);
                a2.subi(a2Velocity);
            }
        }
        for(int i = 0; i < iterations; i++) {
            System.out.println(i);
            if(i%10==0) {
                CostResult[] result = cost(in, theta1, theta2, bias1, bias2);
                System.out.println("Cost "+i+": " + result[1].cost+" --- " + result[1].cost2);
                System.out.println("A1: "+a1.get(0)+" ---- A2: "+a2.get(0));
            }
            double momentum = 0.5;

            ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
            for(int j = 0; j < (input.rows + batchSize-1)/batchSize; j++) {
                int size = Math.min(batchSize, input.rows-j*batchSize);
                DoubleMatrix batch = input.getRange(j*batchSize, j*batchSize+size, 0, input.columns);
                Runnable worker = new StochasticThread(batch, i, input.rows, input.columns, momentum);
                executor.execute(worker);
            }
            executor.shutdown();
            while(!executor.isTerminated());
        }
        CostResult[] result = cost(in, theta1, theta2, bias1, bias2);
        System.out.println("Final Cost: " + result[1].cost);
    }
	
	

	
	private DoubleMatrix[] feedForward(DoubleMatrix patches, DoubleMatrix theta1, DoubleMatrix theta2, DoubleMatrix bias1, DoubleMatrix bias2) {
		DoubleMatrix[] result = new DoubleMatrix[4];
		//z2
		result[0] = patches.mmul(theta1);
		DoubleMatrix bias = DoubleMatrix.ones(patches.rows,1);
		bias = bias.mmul(bias1);
		result[0].addi(bias);
		//a2
		result[1] = Utils.activationFunction(layer1, result[0], a1.get(0));
		//z3
		bias = DoubleMatrix.ones(patches.rows, 1);
		bias = bias.mmul(bias2);
		result[2] = result[1].mmul(theta2);
		result[2].addi(bias);
		//a3
		result[3] = Utils.activationFunction(layer2, result[2], a2.get(0));
		return result;
	}

    public DataContainer feedForward(DataContainer input) {
        return compute(input);
    }

	
	public void writeTheta(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			for(int i = 0; i < theta1.length; i++){
				if( i < theta1.length-1)
					writer.write(theta1.data[i]+",");
				else writer.write(""+theta1.data[i]);
			}
			writer.write('\n');
			for(double d : bias1.data){
				writer.write(d+",");
			}
			writer.close();
			System.out.println(theta1.columns+":"+theta1.rows);
			//visualize(patchRows, patchColumns, (int)Math.sqrt(theta1.columns), filename.replace(".csv", ".png"));
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
			theta1 = new DoubleMatrix(Integer.parseInt(line[0]), Integer.parseInt(line[1]));
			line = reader.readLine().split(",");
			for(int i = 0; i < theta1.rows; i++) {
				for(int j = 0; j < theta1.columns; j++) {
					theta1.put(i, j, Double.parseDouble(line[i * theta1.columns + j]));
				}
			}
			line = reader.readLine().split(",");
			for(int i = 0; i < bias1.rows; i++) {
				for(int j = 0; j < bias1.columns; j++) {
					bias1.put(i, j, Double.parseDouble(line[i * bias1.columns + j]));
				}
			}
            a1.put(0,Double.parseDouble(reader.readLine()));
            a2.put(0,Double.parseDouble(reader.readLine()));
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
			writer.write(theta1.rows+","+theta1.columns+"\n");
			for(int i = 0; i < theta1.rows; i++) {
				for(int j = 0; j < theta1.columns; j++) {
					writer.write(theta1.get(i,j)+",");
				}
			}
			writer.write('\n'+bias1.rows+","+bias1.columns+"\n");
			for(int i = 0; i < bias1.rows; i++) {
				for(int j = 0; j < bias1.columns; j++) {
					writer.write(bias1.get(i,j)+",");
				}
			}
            writer.write(a1.get(0)+"\n");
            writer.write(a2.get(0)+"\n");
			writer.close();
			//visualize(patchRows, patchColumns, (int)Math.sqrt(theta1.columns), filename.replace(".csv", ".png"));
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public DataContainer compute(DataContainer input) {
		DoubleMatrix result = input.getData().mmul(theta1);
		result.addiRowVector(bias1);
        return new DataContainer(Utils.activationFunction(layer1, result));
	}

	public void visualize(int rows, int columns, int images, String filename) throws IOException {
		BufferedImage image = new BufferedImage(columns*images+images*2+2, rows*images+images*2+2, BufferedImage.TYPE_INT_RGB);
		DoubleMatrix tht1 = theta1.dup();
		tht1.subi(tht1.min());
		tht1.divi(tht1.max());
		tht1.muli(255);
		for(int k = 0; k < images; k++) {
			for(int l = 0; l < images; l++) {
				if(k*images+l < tht1.columns) { 
					DoubleMatrix row = tht1.getColumn(k*images+l);
					int channelSize = row.length/3;
					double[] r = new double[channelSize];
					double[] g = new double[channelSize];
					double[] b = new double[channelSize];
					System.arraycopy(row.data, 0, r, 0, channelSize);
					System.arraycopy(row.data, channelSize, g, 0, channelSize);
					System.arraycopy(row.data, 2*channelSize, b, 0, channelSize);
					for(int i = 0; i < rows; i++) {
						for(int j = 0; j < columns; j++) {
							int col = ((int)r[i*columns+j] << 16) | ((int)g[i*columns+j] << 8) | (int)b[i*columns+j];
							image.setRGB(l*(columns+2)+2+j, k*(rows+2)+2+i, col);
						}
					}
				}
			}
		}
		File imageFile = new File(filename);
		ImageIO.write(image, "png", imageFile);
	}

	@Override
	public int domainDimension() {
		return theta1.length+bias1.length+theta2.length+bias2.length+2;
	}

	@Override
	public double valueAt(double[] dTheta) {
		return currentCost[1].cost;
	}

	@Override
	public double[] derivativeAt(double[] dTheta) {
		System.arraycopy(dTheta, 0, theta1.data, 0, theta1.data.length);
		System.arraycopy(dTheta, theta1.data.length, theta2.data, 0, theta2.data.length);
		System.arraycopy(dTheta, theta1.data.length+theta2.data.length, bias1.data, 0, bias1.data.length);
		System.arraycopy(dTheta, theta1.data.length+theta2.data.length+bias1.data.length, bias2.data, 0, bias2.data.length);
        a2.put(0,dTheta[dTheta.length-1]);
        a1.put(0,dTheta[dTheta.length-2]);
		currentCost = cost(new DataContainer(input), theta1, theta2, bias1, bias2);
		double[] derivative = new double[theta1.length+theta2.length+bias1.length+bias2.length+2];
		System.arraycopy(currentCost[0].thetaGrad.data, 0, derivative, 0, theta1.data.length);
		System.arraycopy(currentCost[1].thetaGrad.data, 0, derivative, theta1.data.length, theta2.data.length);
		System.arraycopy(currentCost[0].biasGrad.data, 0, derivative, theta1.data.length+theta2.data.length, bias1.data.length);
		System.arraycopy(currentCost[1].biasGrad.data, 0, derivative, theta1.data.length+theta2.data.length+bias1.data.length, bias2.data.length);
        derivative[derivative.length-1] = currentCost[1].getAGrad();
        derivative[derivative.length-2] = currentCost[0].getAGrad();
		return derivative;
	}

	public DataContainer train(DataContainer input, DoubleMatrix output, int iterations) {
        //alpha = 1.0/(input.rows*input.columns);
        //alpha = 0.01;
        miniBatchGradientDescent(input, iterations, 0.95, 128);
		//lbfgsTrain(input, 200);
        CostResult[] currentCost = cost(input, theta1, theta2, bias1, bias2);
        System.out.println("A1: "+a1.get(0)+" -- A2: " + a2.get(0) + " -- Cost: " +currentCost[1].cost);
        //miniBatchGradientDescent(input, iterations, 0.9, 128);
        try{
            visualize(patchRows, patchColumns, (int)Math.ceil(Math.sqrt(hiddenSize)), "feat"+input.getData().columns+".png");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return compute(input);
	}

}
