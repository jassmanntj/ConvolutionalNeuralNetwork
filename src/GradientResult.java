import org.jblas.DoubleMatrix;

/**
 * Created by jassmanntj on 2/23/2015.
 */
public class GradientResult {
    DoubleMatrix gradient;
    double aGrad;

    public GradientResult(DoubleMatrix gradient, double aGrad) {
        this.gradient = gradient;
        this.aGrad = aGrad;
    }

    public GradientResult(DoubleMatrix gradient) {
        this.gradient = gradient;
        this.aGrad = 0;
    }
}
