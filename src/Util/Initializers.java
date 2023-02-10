package Util;

public class Initializers {
    public interface Supplier {
        double get();
    }

    public static Supplier ZEROS() {
        return () -> 0;
    }

    public static Supplier CONST(double constant) {
        return () -> constant;
    }

    public static Supplier RAND(double amplitude) {
        return () -> (Math.random() * 2 - 1) * amplitude;
    }

    public static Supplier RAND(double amplitude, double offset) {
        return () -> (Math.random() * 2 - 1) * amplitude + offset;
    }
}