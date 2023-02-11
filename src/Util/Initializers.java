package Util;

import java.util.Random;

public class Initializers {
    public interface Supplier {
        double get();
    }

    public static Supplier ZEROS() {
        return () -> 0;
    }

    public static Supplier ONES() {
        return () -> 0;
    }

    public static Supplier CONST(double constant) {
        return () -> constant;
    }

    public static Supplier RAND(double amplitude) {
        return () -> (Math.random() * 2 - 1) * amplitude;
    }

    public static Supplier RAND(double amplitude, double mean) {
        return () -> (Math.random() * 2 - 1) * amplitude + mean;
    }

    public static Supplier GAUSSIAN(double mean, double standardDeviation) {
        Random fRandom = new Random();
        return () -> fRandom.nextGaussian() * standardDeviation + mean;
    }

    /**
     * Weight initializer. Gaussian with mean=0 and standardDeviation=1/sqrt(n)
     * @param n Number of units in the current layer.
     */
    public static Supplier GLOROT(int n) {
        return GAUSSIAN(0, 1 / Math.sqrt(n));
    }
}