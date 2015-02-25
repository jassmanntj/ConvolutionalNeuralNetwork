package cnn.tests;

import cnn.Utils;
import org.jblas.DoubleMatrix;

/**
 * Created by jassmanntj on 2/23/2015.
 */
public class Testing {
    public static void main(String[] args) {
        DoubleMatrix a = new DoubleMatrix(5,5);
        a.put(0,0,5);
        a.put(1,1,234);
        a.put(2,2,-200);
        DoubleMatrix b = Utils.relu(a);
        for(int row = 0; row < b.rows; row++) {
            System.out.println(b.getRow(row));
        }
    }
}
