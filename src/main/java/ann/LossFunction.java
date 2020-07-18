package ann;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public interface LossFunction {

    double evaluateLoss(RealMatrix actual, RealMatrix pred);
    RealMatrix evaluateGradient(RealMatrix actual, RealMatrix pred);
}
