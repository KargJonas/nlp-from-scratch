package util;

import java.io.Serializable;
import java.util.Random;

public class Initializers {
    public interface Supplier extends Serializable {
        float get();
    }

    public static Supplier ZEROS() {
        return () -> 0;
    }

    public static Supplier ONES() {
        return () -> 0;
    }

    public static Supplier CONST(float constant) {
        return () -> constant;
    }

    public static Supplier RAND(float amplitude) {
        return () -> (float) ((Math.random() * 2 - 1) * amplitude);
    }

    public static Supplier RAND(float amplitude, float mean) {
        return () -> (float) ((Math.random() * 2 - 1) * amplitude + mean);
    }

    public static Supplier GAUSSIAN(float mean, float standardDeviation) {
        Random fRandom = new Random();
        return () -> (float) (fRandom.nextGaussian() * standardDeviation + mean);
    }

    /**
     * Kaiming initialization mean=0 and standardDeviation=2/n
     * @param n Number of units
     */
    public static Supplier KAIMING(int n) {
        return GAUSSIAN(0, 2f / n);
    }

    /**
     * Weight initializer. Gaussian with mean=0 and standardDeviation=1/sqrt(n)
     * @param n Number of units in the current layer.
     */
    public static Supplier GLOROT(int n) {
        return GAUSSIAN(0f, (float) (1 / Math.sqrt(n)));
    }
}