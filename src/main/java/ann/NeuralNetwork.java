package ann;

import org.apache.commons.math3.linear.RealMatrix;

public interface NeuralNetwork {

    RealMatrix predict(RealMatrix X);
    void learn(RealMatrix X, RealMatrix actual);

}
