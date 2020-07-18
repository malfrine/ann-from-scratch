package model;

public class Image {

    private Pixel[][] pixels;

    public Image(Pixel[][] pixels) {
        this.pixels = pixels;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        for (Pixel[] pixelRow : pixels) {
            for (Pixel pixel : pixelRow) {
                stringBuilder.append(pixel.getValue()).append(",");
            }
            stringBuilder.append("\n");
        }
        return stringBuilder.toString();
    }

    public Pixel getPixel(int i, int j) {
        return pixels[i][j];
    }

    public int getNumRows() {
        return pixels.length;
    }

    public int getNumColumns() {
        return pixels[0].length;
    }

    public static class Builder{

        private int numRows;
        private int numColumns;
        private Pixel[][] pixels;

        private int curRowIndex = 0;
        private int curColIndex = 0;

        private Builder(int numRows, int numColumns) {
            this.numRows = numRows;
            this.numColumns = numColumns;
            this.pixels = new Pixel[numRows][numColumns];
        }

        public static Builder newInstance(int numRows, int numColumns) {
            return new Builder(numRows, numColumns);
        }

        public Builder addPixel(Pixel pixel) throws IllegalAccessException {

            if (curColIndex < numColumns) {
                pixels[curRowIndex][curColIndex] = pixel;
                curColIndex++;
            } else if (curColIndex == numColumns) {
                if (curRowIndex + 1 >= numRows) {
                    throw new IllegalAccessException("Max rows limit exceeded.");
                }
                curColIndex = 0;
                curRowIndex++;
                pixels[curRowIndex][curColIndex] = pixel;
                curColIndex++;
            } else {
                throw new IllegalAccessException("Max columns limit exceeded.");
            }
            return this;
        }

        public Image build() {
            return new Image(pixels);
        }
    }
}
