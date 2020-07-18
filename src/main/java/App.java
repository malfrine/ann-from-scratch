
import dao.DataReader;
import ann.ArtificialNeuralNetwork;
import model.DataCollector;
import model.DigitClassificationNeuralNetwork;

import java.io.IOException;
import java.util.logging.Logger;


public class App {

    private static final Logger logger = Logger.getLogger(App.class.getName());

    public static void main(String[] args) throws IOException, IllegalAccessException {
        DataReader dataReader = new DataReader();
        DataCollector dataCollector = dataReader.readData(50);

        logger.info(dataCollector.getImages().get(0).toString());
        logger.info(dataCollector.getNumbers().get(0).toString());

        DigitClassificationNeuralNetwork digitClassificationNeuralNetwork =
                new DigitClassificationNeuralNetwork(dataCollector, 5000, 0.1);

        digitClassificationNeuralNetwork.run();

    }

}
