package model;

public class Pixel {

    private double value;

    public Pixel(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Pixel) {
            return value == ((Pixel) obj).getValue();
        }
        return false;
    }
}
