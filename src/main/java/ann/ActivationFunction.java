package ann;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public interface ActivationFunction {
    RealVector evaluate(RealVector x);
    RealMatrix evaluate(RealMatrix x);
    RealVector evaluateGradient(RealVector x);
    RealMatrix evaluateGradient(RealMatrix x);
}
