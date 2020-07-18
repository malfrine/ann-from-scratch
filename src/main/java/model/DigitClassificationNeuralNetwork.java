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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DigitClassificationNeuralNetwork {

    private DataCollector dataCollector;
    private Random rng = new Random();

    public DigitClassificationNeuralNetwork(DataCollector dataCollector) {
        this.dataCollector = dataCollector;
    }

    public void run() throws IOException, IllegalAccessException {
        RealMatrix inputs = makeInputs();
        RealMatrix outputs = makeOutputs();

        System.out.println("input row size: " + inputs.getRowDimension());
        System.out.println("input columns size: " + inputs.getColumnDimension());
        System.out.println("output rows size: " + outputs.getRowDimension());
        System.out.println("output columns size: " + outputs.getColumnDimension());

        ArtificialNeuralNetwork.Builder annBuilder = ArtificialNeuralNetwork.Builder.newInstance(new NormalLossFunction());
        annBuilder.addLayer(inputs.getRowDimension(), new SigmoidFunction());
        annBuilder.addLayer(30, new SigmoidFunction());
        annBuilder.addFinalLayer(outputs.getRowDimension());
        ArtificialNeuralNetwork ann = annBuilder.build();

        RealMatrix firstPrediction = ann.predict(inputs);
        System.out.println(inputs.transpose());
        System.out.println(outputs.transpose());
        System.out.println(firstPrediction.transpose());

        int numIterations = 5000;
        for (int i = 0; i < numIterations; i++) {
            ann.learn(inputs, outputs);
            RealMatrix prediction = ann.predict(inputs);
            int randomIndex = rng.nextInt(inputs.getColumnDimension());
            ann.learnOnSingle(inputs.getColumnVector(randomIndex), outputs.getColumnVector(randomIndex));
            if (i % (numIterations / 100) == 0) {
                double accuracy = calculateAccuracy(prediction);
                System.out.println(
                        "Iterations: " + i
                        + "; accuracy: " + accuracy * 100 + "%"
                        + "; loss: " + ann.getLossFunction().evaluateLoss(outputs, prediction
                        )
                );
            }

        }
        RealMatrix finalPrediction = ann.predict(inputs);

        int numImages = dataCollector.getNumbers().size();
        int numAccuratePredictions = 0;
        for (int i = 0; i < numImages; i++) {
            int actualNumber = dataCollector.getNumbers().get(i);
            int predictedNumber = finalPrediction.getColumnVector(i).getMaxIndex();
            double confidence = finalPrediction.getColumnVector(i).getMaxValue();
            if (actualNumber == predictedNumber) {
                numAccuratePredictions++;
            }
            System.out.println("Actual: " + actualNumber + "; Predicted: "
                    + predictedNumber + " with confidence: " + confidence + "%");
        }
        double accuracy = (double) numAccuratePredictions / numImages;
        System.out.println("Model accuracy: " + accuracy * 100 + "%");


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
        int numAccuratePredictions = 0;
        for (int i = 0; i < numImages; i++) {
            int actualNumber = dataCollector.getNumbers().get(i);
            int predictedNumber = prediction.getColumnVector(i).getMaxIndex();
            // double confidence = prediction.getColumnVector(i).getMaxValue();
            if (actualNumber == predictedNumber) {
                numAccuratePredictions++;
            }
//            System.out.println("Actual: " + actualNumber + "; Predicted: "
//                    + predictedNumber + " with confidence: " + confidence + "%");
        }
        return (double) numAccuratePredictions / numImages;
        // System.out.println("Model accuracy: " + accuracy * 100 + "%");
    }

}
