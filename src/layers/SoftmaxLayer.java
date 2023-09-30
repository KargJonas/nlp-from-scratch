package layers;

import util.LayerType;

public class SoftmaxLayer extends GenericLayer<SoftmaxLayer> {
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
        activations = new float[getSize()];
    }

    public static SoftmaxLayer build(int size) {
        return new SoftmaxLayer(size);
    }

    @Override
    public void computeActivations() {
        if (parentLayer == null) return;

        float[] parentLayerActivations = parentLayer.getActivations();
        float softmaxCoefficient = 0;

        for (int j = 0; j < parentLayer.getSize(); j++) {
            softmaxCoefficient += Math.pow(Math.E, parentLayerActivations[j]);
        }

        for (int i = 0; i < getSize(); i++) {
            activations[i] = (float) (Math.pow(Math.E, parentLayerActivations[i]) / softmaxCoefficient);
        }
    }

    @Override
    public void backprop(float[] errors, float learningRate) {
        // No need to update weights and biases for this layer as there are none.

        float[] parentError = new float[parentLayer.getSize()];

        for (int i = 0; i < getSize(); i++) {
            for (int j = 0; j < parentLayer.getSize(); j++) {
                // Cross-product of errors with Jacobian.
                parentError[j] += errors[i] * (i == j ? activations[i] * (1 - activations[i]) : -activations[i] * activations[j]);
            }
        }

        parentLayer.backprop(parentError, learningRate);
    }
}
