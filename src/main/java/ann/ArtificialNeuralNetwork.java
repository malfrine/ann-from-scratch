package ann;


import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;

public class ArtificialNeuralNetwork {

    private static final double RATE = 0.05 ;

    private List<Layer> layers;
    private LossFunction lossFunction;

    public ArtificialNeuralNetwork(List<Layer> layers, LossFunction lossFunction) {
        this.layers = layers;
        this.lossFunction = lossFunction;
    }

    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public RealMatrix predict(RealMatrix X) {
        RealMatrix nextLayerInput = X;
        for (Layer layer : layers) {
            nextLayerInput = layer.getOutput(nextLayerInput);
        }
        return nextLayerInput;
    }

    public void learn(RealMatrix X, RealMatrix actual) {
        // perform feed-forward and record output at each layer
        List<RealMatrix> layerInputs = new ArrayList<>();
        RealMatrix layerInput = X;
        for (Layer layer : layers) {
            layerInputs.add(layerInput);
            layerInput = layer.getOutput(layerInput);
        }
        int numLayers = layers.size();
        RealMatrix lossChangeDueToOutput = lossFunction.evaluateGradient(actual, layerInput);
        for (int i = numLayers - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            layerInput = layerInputs.get(i);
            RealMatrix activationFunctionInput = layer.dotProductOnWeights(layerInput);
            RealMatrix outputAndActivationDerivatives = Utilities.ebeMultiply(
                    layer.activationFunction.evaluateGradient(activationFunctionInput),
                    lossChangeDueToOutput
            );
            RealMatrix lossChangeDueToLayerWeights = layerInput.multiply(outputAndActivationDerivatives.transpose());
            lossChangeDueToOutput = layer.outputWeights.multiply(outputAndActivationDerivatives);
            layer.outputWeights = layer.outputWeights.subtract(lossChangeDueToLayerWeights.scalarMultiply(RATE));
        }
    }

    private static class Layer {
        RealMatrix outputWeights;
        ActivationFunction activationFunction;

        public Layer(RealMatrix startingWeights, ActivationFunction activationFunction) {
            this.outputWeights = startingWeights;
            this.activationFunction = activationFunction;
        }

        private RealMatrix dotProductOnWeights(RealMatrix input) {
            return outputWeights.transpose().multiply(input);
        }

        private RealMatrix getOutput(RealMatrix input) {
            return activationFunction.evaluate(dotProductOnWeights(input));
        }
    }


    public static class Builder {

        private ArrayList<Integer> numNeurons;
        private ArrayList<ActivationFunction> activationFunctions;
        private LossFunction lossFunction;

        private Builder(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            this.numNeurons = new ArrayList<>();
            this.activationFunctions = new ArrayList<>();
        }

        public static Builder newInstance(LossFunction lossFunction) {
            return new Builder(lossFunction);
        }

        public Builder addLayer(int numNeurons, ActivationFunction activationFunction) {
            this.numNeurons.add(numNeurons);
            this.activationFunctions.add(activationFunction);
            return this;
        }

        public Builder addFinalLayer(int numNeurons) {
            this.numNeurons.add(numNeurons);
            this.activationFunctions.add(null);
            return this;
        }

        private List<Layer> makeLayers() {
            ArrayList<Layer> layers = new ArrayList<>();
            for (int i = 0; i < numNeurons.size() - 1; i++) {
                int curLayerSize = numNeurons.get(i);
                int nextLayerSize = numNeurons.get(i+1);
                RealMatrix startingWeights = Utilities.generateRandomRealMatrix(curLayerSize, nextLayerSize);
                layers.add(new Layer(startingWeights, activationFunctions.get(i)));
            }
            return layers;
        }

        public ArtificialNeuralNetwork build() {
            List<Layer> layers = makeLayers();
            return new ArtificialNeuralNetwork(layers, lossFunction);
        }

    }

}
