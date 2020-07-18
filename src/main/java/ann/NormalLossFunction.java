package ann;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.stream.IntStream;

public class NormalLossFunction implements LossFunction {

    @Override
    public double evaluateLoss(RealVector actual, RealVector pred) {
        RealVector diff = pred.subtract(actual);
        return diff.dotProduct(diff);
    }

    @Override
    public double evaluateLoss(RealMatrix actual, RealMatrix pred) {
        RealMatrix diff = pred.subtract(actual);
        return IntStream.range(0, diff.getRowDimension())
                .parallel()
                .mapToDouble(i -> diff.getRowVector(i).getL1Norm())
                .sum();
    }

    @Override
    public RealVector evaluateGradient(RealVector actual, RealVector pred) {
        return pred.subtract(actual);
    }

    @Override
    public RealMatrix evaluateGradient(RealMatrix actual, RealMatrix pred) {
        return pred.subtract(actual);
    }
}
