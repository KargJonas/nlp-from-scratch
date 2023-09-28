package util;

public class Activations {
    public interface ActivationFn {
        float f(float x);

        float df(float x);
    }

    public static ActivationFn IDENTITY = new ActivationFn() {
        public float f(float x) {
            return x;
        }

        public float df(float x) {
            return 1.0f;
        }
    };

    public static ActivationFn BIN_STEP = new ActivationFn() {
        public float f(float x) {
            return x < 0 ? 0 : 1;
        }

        public float df(float x) {
            return 0.0f;
        }
    };

    public static ActivationFn LOGISTIC = new ActivationFn() {
        public float f(float x) {
            return (float) (1 / (1 + Math.pow(Math.E, -x)));
        }

        public float df(float x) {
            float fx = f(x);
            return fx * (1 - fx);
        }
    };

    public static ActivationFn TANH = new ActivationFn() {
        public float f(float x) {
            return (float) Math.tanh(x);
        }

        public float df(float x) {
            return (float) (1 - Math.pow(f(x), 2));
        }
    };

    public static ActivationFn RELU = new ActivationFn() {
        public float f(float x) {
            return x > 0 ? x : 0;
        }

        public float df(float x) {
            return x > 0 ? 1.0f : 0.0f;
        }
    };

    public static ActivationFn SOFTPLUS = new ActivationFn() {
        public float f(float x) {
            return (float) Math.log(1 + Math.pow(Math.E, x));
        }

        public float df(float x) {
            return (float) (1 / (1 + Math.pow(Math.E, -x)));
        }
    };

    public static ActivationFn SELU = new ActivationFn() {
        public float f(float x) {
            return (float) (1.0507 * (x < 0 ? 1.67326 * (Math.pow(Math.E, x) - 1) : x));
        }

        public float df(float x) {
            return x < 0
                    ? (float) (1.7581 * Math.pow(Math.E, x))
                    : 1.0507f;
        }
    };

    public static ActivationFn LRELU = new ActivationFn() {
        public float f(float x) {
            return x < 0 ? 0.01f * x : x;
        }

        public float df(float x) {
            return x < 0 ? 0.01f : 1.0f;
        }
    };

    public static ActivationFn SIGMOID = new ActivationFn() {
        public float f(float x) {
            return (float) (x / (1 + Math.pow(Math.E, -x)));
        }

        public float df(float x) {
            float factor = (float) Math.pow(Math.E, -x);
            return (float) ((1 + factor + x * factor) / Math.pow(1 + factor, 2));
        }
    };

    public static ActivationFn GAUSSIAN = new ActivationFn() {
        public float f(float x) {
            return (float) Math.pow(Math.E, -Math.pow(x, 2));
        }

        public float df(float x) {
            return (float) (-2 * x * Math.pow(Math.E, -Math.pow(x, 2)));
        }
    };

//    public static ActivationFn ELU(float a) {
//        return (x) -> x > 0 ? x : a * (Math.pow(Math.E, x) - 1);
//    }
//    public static ActivationFn PRLU(float a) {
//        return (x) -> x < 0 ? a * x : x;
//    }

//    public static ActivationFn ELU = new ActivationFn() {
//        public float f(float a) {
//            return (x) -> x > 0 ? x : a * (Math.pow(Math.E, x) - 1);
//        }
//
//        public float df(float a) {
//            return (x) -> x > 0 ? 1 : a * Math.pow(Math.E, x);
//        }
//    }
//
//    public static ParametricActivationFn PRLU = new ParametricActivationFn() {
//        public ActivationFn f(float a) {
//            return (x) -> x < 0 ? a * x : x;
//        }
//
//        public ActivationFn df(float a) {
//            return (x) -> x < 0 ? a : 1;
//        }
//    };
}
