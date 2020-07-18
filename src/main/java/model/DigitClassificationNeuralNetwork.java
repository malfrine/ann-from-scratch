package model;

import ann.ArtificialNeuralNetwork;
import ann.NormalLossFunction;
import ann.SigmoidFunction;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DigitClassificationNeuralNetwork {

    private static final Logger logger = Logger.getLogger(DigitClassificationNeuralNetwork.class.getName());
    private static final Random rng = new Random();

    private final DataCollector dataCollector;
    private final int numIterations;
    private final double testTrainSplit;

    public DigitClassificationNeuralNetwork(
            DataCollector dataCollector,
            int numIterations,
            double testTrainSplit
    ) {
        if (numIterations < 0) {
            throw new IllegalArgumentException("number of iterations must be greater than or equal to 0");
        }
        if (testTrainSplit > 1) {
            throw new IllegalArgumentException("test train split factor must be less than 1.");
        }
        this.dataCollector = dataCollector;
        this.numIterations = numIterations;
        this.testTrainSplit = testTrainSplit;
    }

    public void run() throws IOException, IllegalAccessException {
        RealMatrix inputs = makeInputs();
        RealMatrix outputs = makeOutputs();

        logger.info("input row size: " + inputs.getRowDimension());
        logger.info("input columns size: " + inputs.getColumnDimension());
        logger.info("output rows size: " + outputs.getRowDimension());
        logger.info("output columns size: " + outputs.getColumnDimension());

        ArtificialNeuralNetwork ann = buildAnn(inputs, outputs);
        runGradientDescent(ann, inputs, outputs);

        RealMatrix finalPrediction = ann.predict(inputs);
        logEachPrediction(finalPrediction);
        logger.info("Final accuracy: " + calculateAccuracy(finalPrediction) * 100 + "%");
    }

    private void runGradientDescent(ArtificialNeuralNetwork ann, RealMatrix inputs, RealMatrix outputs) {
        for (int i = 0; i < numIterations; i++) {
            ann.learn(inputs, outputs);
            RealMatrix prediction = ann.predict(inputs);
            int randomIndex = rng.nextInt(inputs.getColumnDimension());
            ann.learn(inputs.getColumnMatrix(randomIndex), outputs.getColumnMatrix(randomIndex));
            if (i % (numIterations / 100) == 0) {
                double accuracy = calculateAccuracy(prediction);
                logger.info(
                        "Iterations: " + i
                                + "; accuracy: " + accuracy * 100 + "%"
                                + "; loss: " + ann.getLossFunction().evaluateLoss(outputs, prediction
                        )
                );
            }
        }
    }

    private ArtificialNeuralNetwork buildAnn(RealMatrix inputs, RealMatrix outputs) {
        ArtificialNeuralNetwork.Builder annBuilder = ArtificialNeuralNetwork.Builder.newInstance(new NormalLossFunction());
        annBuilder.addLayer(inputs.getRowDimension(), new SigmoidFunction());
        annBuilder.addLayer(30, new SigmoidFunction());
        annBuilder.addFinalLayer(outputs.getRowDimension());
        return annBuilder.build();
    }


    private RealMatrix makeInputs() {
        int numPixels = dataCollector.getImages().get(0).getNumRows()
                * dataCollector.getImages().get(0).getNumColumns();
        double[][] inputs = new double[numPixels + 1][dataCollector.getImages().size()];
        int imageIndex = 0;
        for (Image image : dataCollector.getImages()) {
            for (int i = 0; i < image.getNumRows(); i++) {
                for (int j = 0; j < image.getNumColumns(); j++) {
                    int n = i * (image.getNumRows() - 1) + j;
                    inputs[n][imageIndex] = image.getPixel(i, j).getValue();
                }
            }
            inputs[numPixels][imageIndex] = 1;// bias term
            imageIndex++;
        }
        RealMatrix rawInputs = new Array2DRowRealMatrix(inputs);
        RealMatrix normalizedInputs = new Array2DRowRealMatrix(rawInputs.getRowDimension(),rawInputs.getColumnDimension());
                IntStream.range(0, imageIndex).parallel().forEach( i ->
                        normalizedInputs.setColumn(i, StatUtils.normalize(rawInputs.getColumnVector(i).toArray()))
        );
        return normalizedInputs;
    }

    private RealMatrix makeOutputs() {
        List<Integer> uniqueIntegers = new HashSet<>(dataCollector.getNumbers())
                .stream().sorted().collect(Collectors.toList());
        int numIndex = 0;
        double[][] outputs = new double[uniqueIntegers.size()][dataCollector.getNumbers().size()];
        for (Integer number : dataCollector.getNumbers()) {
            for (int i = 0; i < uniqueIntegers.size(); i++) {
                outputs[i][numIndex] = number.equals(uniqueIntegers.get(i)) ? 1 : 0;
            }
            numIndex++;
        }
        return new Array2DRowRealMatrix(outputs);
    }

    private double calculateAccuracy(RealMatrix prediction) {
        int numImages = dataCollector.getNumbers().size();
        return IntStream.range(0, numImages).mapToDouble(i ->
                dataCollector.getNumbers().get(i) == prediction.getColumnVector(i).getMaxIndex() ? 1 : 0
        ).sum()  / numImages;
    }

    private void logEachPrediction(RealMatrix prediction) {
        for (int i = 0; i < dataCollector.getImages().size(); i++) {
            logger.info(
                    "Actual: " + dataCollector.getNumbers().get(i)
                            + "; Predicted: " + prediction.getColumnVector(i).getMaxIndex()
                            + " with confidence: " + prediction.getColumnVector(i).getMaxValue() * 100 + "%");
        }
    }

}
