
import dao.DataReader;
import model.DataCollector;
import model.DigitClassificationNeuralNetwork;

import java.io.IOException;
import java.util.logging.Logger;


public class App {

    private static final Logger logger = Logger.getLogger(App.class.getName());
    private static final int NUM_IMAGES = 500;
    private static final int NUM_ITERATIONS = 5000;

    public static void main(String[] args) throws IOException, IllegalAccessException {
        DataReader dataReader = new DataReader();
        DataCollector dataCollector = dataReader.readData(NUM_IMAGES);
        DigitClassificationNeuralNetwork digitClassificationNeuralNetwork =
                new DigitClassificationNeuralNetwork(dataCollector, NUM_ITERATIONS);
        digitClassificationNeuralNetwork.run();
    }

}
