package layers;

import util.LayerType;

public class SoftmaxLayer extends Layer<SoftmaxLayer> {
    SoftmaxLayer(int size) {
        super(size);
        layerType = LayerType.SOFTMAX;
    }

    @Override
    protected SoftmaxLayer getThis() {
        return null;
    }

    @Override
    public void initialize() {
        // No need to initialize
        activations = new double[getSize()];
    }

    @Override
    public void initializeValues() { }

    public static SoftmaxLayer build(int size) {
        return new SoftmaxLayer(size);
    }

    @Override
    public void computeActivations() {
        if (parentLayer == null) return;

        double[] parentLayerActivations = parentLayer.getActivations();
        double softmaxCoefficient = 0;

        for (int j = 0; j < parentLayer.getSize(); j++) {
            softmaxCoefficient += Math.pow(Math.E, parentLayerActivations[j]);
        }

        for (int i = 0; i < getSize(); i++) {
            activations[i] = Math.pow(Math.E, parentLayerActivations[i]) / softmaxCoefficient;
        }
    }

    @Override
    public void backprop(double[] errors, double learningRate) {
        // No need to update weights and biases for this layer as there are none.

        double[] parentError = new double[parentLayer.getSize()];

        for (int i = 0; i < getSize(); i++) {
            for (int j = 0; j < parentLayer.getSize(); j++) {
                // Cross-product of errors with Jacobian.
                parentError[j] += errors[i] * (i == j ? activations[i] * (1 - activations[i]) : -activations[i] * activations[j]);
            }
        }

        parentLayer.backprop(parentError, learningRate);
    }
}
