package cnn.tests;


import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jtransforms.fft.DoubleFFT_2D;
import org.junit.Test;

public class TestFFT {

	@Test
	public void test() {
		DoubleMatrix input = DoubleMatrix.rand(4, 4);
		DenseDoubleAlgebra d = new DenseDoubleAlgebra();
		
		DenseDoubleMatrix2D in = new DenseDoubleMatrix2D(input.toArray2());
		ComplexDoubleMatrix cd = input.toComplex();
		DoubleFFT_2D t = new DoubleFFT_2D(input.rows, input.columns);
		DenseDComplexMatrix2D dc = in.getFft2();
		//in.fft2();
		t.complexForward(cd.data);
		for(int i = 0; i < cd.rows; i++) {
			System.out.println(cd.getRow(i));
		}
		System.out.println("\n"+dc);
		
		/*System.out.println(dc.get(0, 0)[0]+":"+cd.get(0,0).real());
		System.out.println(dc.get(0, 0)[1]+":"+cd.get(0,0).imag());	
		System.out.println(dc.get(0, 1)[0]+":"+cd.get(0,1).real());
		System.out.println(dc.get(0, 1)[1]+":"+cd.get(0,1).imag());
		System.out.println(dc.get(1, 0)[0]+":"+cd.get(1,0).real());
		System.out.println(dc.get(1, 0)[1]+":"+cd.get(1,0).imag());*/
	}

}
