package ann;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public interface ActivationFunction {
    RealMatrix evaluate(RealMatrix x);
    RealMatrix evaluateGradient(RealMatrix x);
}
