package Util;

public class Activations {
    public interface ActivationFn {
        double f(double x);

        double df(double x);
    }

    public static ActivationFn IDENTITY = new ActivationFn() {
        public double f(double x) {
            return x;
        }

        public double df(double x) {
            return 1.0;
        }
    };

    public static ActivationFn BIN_STEP = new ActivationFn() {
        public double f(double x) {
            return x < 0 ? 0 : 1;
        }

        public double df(double x) {
            return 0.0;
        }
    };

    public static ActivationFn LOGISTIC = new ActivationFn() {
        public double f(double x) {
            return 1 / (1 + Math.pow(Math.E, -x));
        }

        public double df(double x) {
            double fx = f(x);
            return fx * (1 - fx);
        }
    };

    public static ActivationFn TANH = new ActivationFn() {
        public double f(double x) {
            return Math.tanh(x);
        }

        public double df(double x) {
            return 1 - Math.pow(f(x), 2);
        }
    };

    public static ActivationFn RELU = new ActivationFn() {
        public double f(double x) {
            return x > 0 ? x : 0;
        }

        public double df(double x) {
            return x > 0 ? 1.0 : 0.0;
        }
    };

    public static ActivationFn SOFTPLUS = new ActivationFn() {
        public double f(double x) {
            return Math.log(1 + Math.pow(Math.E, x));
        }

        public double df(double x) {
            return 1 / (1 + Math.pow(Math.E, -x));
        }
    };

    public static ActivationFn SELU = new ActivationFn() {
        public double f(double x) {
            return 1.0507 * (x < 0 ? 1.67326 * (Math.pow(Math.E, x) - 1) : x);
        }

        public double df(double x) {
            return x < 0
                    ? 1.7581 * Math.pow(Math.E, x)
                    : 1.0507;
        }
    };

    public static ActivationFn LRELU = new ActivationFn() {
        public double f(double x) {
            return x < 0 ? 0.01 * x : x;
        }

        public double df(double x) {
            return x < 0 ? 0.01 : 1.0;
        }
    };

    public static ActivationFn SIGMOID = new ActivationFn() {
        public double f(double x) {
            return x / (1 + Math.pow(Math.E, -x));
        }

        public double df(double x) {
            double factor = Math.pow(Math.E, -x);
            return (1 + factor + x * factor) / Math.pow(1 + factor, 2);
        }
    };

    public static ActivationFn GAUSSIAN = new ActivationFn() {
        public double f(double x) {
            return Math.pow(Math.E, -Math.pow(x, 2));
        }

        public double df(double x) {
            return -2 * x * Math.pow(Math.E, -Math.pow(x, 2));
        }
    };

//    public static ActivationFn ELU(double a) {
//        return (x) -> x > 0 ? x : a * (Math.pow(Math.E, x) - 1);
//    }
//    public static ActivationFn PRLU(double a) {
//        return (x) -> x < 0 ? a * x : x;
//    }

//    public static ActivationFn ELU = new ActivationFn() {
//        public double f(double a) {
//            return (x) -> x > 0 ? x : a * (Math.pow(Math.E, x) - 1);
//        }
//
//        public double df(double a) {
//            return (x) -> x > 0 ? 1 : a * Math.pow(Math.E, x);
//        }
//    }
//
//    public static ParametricActivationFn PRLU = new ParametricActivationFn() {
//        public ActivationFn f(double a) {
//            return (x) -> x < 0 ? a * x : x;
//        }
//
//        public ActivationFn df(double a) {
//            return (x) -> x < 0 ? a : 1;
//        }
//    };
}
