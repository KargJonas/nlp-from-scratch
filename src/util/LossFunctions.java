package util;

public class LossFunctions {
    public interface LossFn {
        double f(double[] prediction, double[] truth);
    }

    public static LossFn CATEGORICAL_CROSSENTROPY = (x, y) -> {
        double crossEntropy = 0;

        for (int i = 0; i < y.length; i++) {
            crossEntropy -= y[i] * Math.log(x[i]);
        }

        return crossEntropy;
    };

    public static LossFn MEAN_SQUARED_ERROR = (x, y) -> {
        double sum = 0;

        for (int i = 0; i < y.length; i++) {
            sum += Math.pow(x[i] - y[i], 2);
        }

        return sum;
    };
}
