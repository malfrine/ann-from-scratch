package dao;

import model.DataCollector;
import model.Image;
import model.Pixel;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DataReader {

    private static final int LINE_LENGTH = 28;

    public DataCollector readData(int numImages) throws IOException, IllegalAccessException {

        List<Integer> numbers = new ArrayList<>();
        List<Image> images = new ArrayList<>();

        File file = new File("data/mnist_train.csv");
        Scanner scanner = new Scanner(new FileInputStream(file));
        int imageCounter = 0;
        while (scanner.hasNext() && imageCounter < numImages) {
            String[] line = scanner.next().split(",");
            numbers.add(Integer.parseInt(line[0]));
            Image.Builder imageBuilder = Image.Builder.newInstance(28, 28);
            for (int j = 1; j < line.length; j++) {
                double intensity = Double.parseDouble(line[j]);
                imageBuilder.addPixel(new Pixel(intensity));
            }
            images.add(imageBuilder.build());
            imageCounter++;
        }
        scanner.close();
        return new DataCollector(images, numbers);
    }



}
