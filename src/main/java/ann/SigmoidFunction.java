package ann;

import org.apache.commons.math3.linear.*;

public class SigmoidFunction implements ActivationFunction {
    @Override
    public RealVector evaluate(RealVector x) {
        return x.map(this::evaluate);
    }

    @Override
    public RealVector evaluateGradient(RealVector x) {
        return x.map(this::evaluateGradient);
    }

    @Override
    public RealMatrix evaluate(RealMatrix x) {
        return Utilities.applyToEachRow(x, this::evaluate);
    }

    @Override
    public RealMatrix evaluateGradient(RealMatrix x) {
        return Utilities.applyToEachRow(x, this::evaluateGradient);
    }

    private double evaluate(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double evaluateGradient(double x) {
        double y = evaluate(x);
        return y * (1 - y);
    }

}
