package Util;

public class Activation {
    public interface ActivationFn {
        double f(double x);
    }

    public static ActivationFn IDENTITY = (x) -> x;
    public static ActivationFn BIN_STEP = (x) -> x < 0 ? 0 : 1;
    public static ActivationFn LOGISTIC = (x) -> 1 / (1 + Math.pow(Math.E, -x));
    public static ActivationFn TANH = Math::tanh;
    public static ActivationFn RELU = (x) -> x > 0 ? x : 0;
    public static ActivationFn SOFTPLUS = (x) -> Math.log(1 + Math.pow(Math.E, x));
    public static ActivationFn SELU = (x) -> 1.0507 * (x < 0 ? 1.67326 * (Math.pow(Math.E, x) - 1) : x);
    public static ActivationFn LRELU = (x) -> x < 0 ? 0.01 * x : x;
    public static ActivationFn SIGMOID = (x) -> x / (1 + Math.pow(Math.E, -x));
    public static ActivationFn GAUSSIAN =  (x) -> Math.pow(Math.E, -Math.pow(x, 2));
    public static ActivationFn ELU(double a) {
        return (x) -> x > 0 ? x : a * (Math.pow(Math.E, x) - 1);
    }
    public static ActivationFn PRLU(double a) {
        return (x) -> x < 0 ? a * x : x;
    }
}