package model;

import java.util.List;

public class DataCollector {

    private final List<Image> images;
    private final List<Integer> numbers;

    public DataCollector(List<Image> images, List<Integer> numbers) {
        this.images = images;
        this.numbers = numbers;
    }

    public List<Image> getImages() {
        return images;
    }

    public List<Integer> getNumbers() {
        return numbers;
    }
}
