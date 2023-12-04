//package layers;
//
//public class DenseLayer extends LayerDecorator {
//    protected DenseLayer(ILayer baseLayer) {
//        super(baseLayer);
//    }
//
//    /**
//     * Computes the activations based on weights, biases and the previous layers' activations.
//     */
//    @Override
//    public void computeActivations() {
//        IBasicLayer parentLayer = getParentLayer();
//        if (parentLayer == null) return;
//
//        for (int i = 0; i < getSize(); i++) {
//            float weightedSum = 0;
//
//            for (int j = 0; j < parentLayer.getSize(); j++) {
//                weightedSum += parentLayer.getActivations()[j] * getWeights()[i][j];
//            }
//
//            getActivations()[i] = getActivationFn().f(weightedSum + getBiases()[i]);
//        }
//    }
//
//    public static DenseLayer build(int size) {
//        return new DenseLayer(Layer.build(size));
//    }
//}


package layers;

import util.Initializers;

public class DenseLayer extends LayerDecorator {
    protected DenseLayer(ILayer baseLayer) {
        super(baseLayer);
    }

    /**
     * Computes the activations based on weights, biases, and the previous layers' activations.
     */
    @Override
    public void computeActivations() {
        IBasicLayer parentLayer = getParentLayer();
        if (parentLayer == null) return;

        float[] logits = new float[getSize()]; // Store the raw logits

        for (int i = 0; i < getSize(); i++) {
            float weightedSum = 0;

            for (int j = 0; j < parentLayer.getSize(); j++) {
                weightedSum += parentLayer.getActivations()[j] * getWeights()[i][j];
            }

            // Update the logits (raw values) before applying softmax
            logits[i] = weightedSum + getBiases()[i];
        }

        // Apply the softmax activation function to compute probabilities
        float[] activations = softmax(logits);
        setActivations(activations);
    }

    // Softmax activation function
    private float[] softmax(float[] logits) {
        float[] probabilities = new float[logits.length];
        float maxLogit = Float.NEGATIVE_INFINITY;

        // Find the maximum logit to prevent overflow
        for (float logit : logits) {
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }

        float sumExp = 0;

        // Compute the sum of exponentials for normalization
        for (int i = 0; i < logits.length; i++) {
            sumExp += Math.exp(logits[i] - maxLogit);
        }

        // Calculate softmax probabilities
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = (float) (Math.exp(logits[i] - maxLogit) / sumExp);
        }

        return probabilities;
    }

    public static DenseLayer build(int size) {
        return new DenseLayer(Layer.build(size));
    }
}
