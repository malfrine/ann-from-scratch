package ann;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public interface LossFunction {

    double evaluateLoss(RealVector actual, RealVector pred);
    double evaluateLoss(RealMatrix actual, RealMatrix pred);
    RealVector evaluateGradient(RealVector actual, RealVector pred);
    RealMatrix evaluateGradient(RealMatrix actual, RealMatrix pred);
}
