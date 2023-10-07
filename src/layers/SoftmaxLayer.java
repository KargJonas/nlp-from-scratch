package layers;

import util.LayerType;

public class SoftmaxLayer extends LayerDecorator {
    SoftmaxLayer(ILayer size) {
        super(size);
    }

    @Override
    public void initialize() {
        // No need to initialize
        setActivations(new float[getSize()]);
    }

    @Override
    public void computeActivations() {
        IBasicLayer parentLayer = getParentLayer();

        if (parentLayer == null) return;

        float[] parentLayerActivations = parentLayer.getActivations();
        float softmaxCoefficient = 0;

        for (int j = 0; j < parentLayer.getSize(); j++) {
            softmaxCoefficient += (float) Math.pow(Math.E, parentLayerActivations[j]);
        }

        for (int i = 0; i < getSize(); i++) {
            getActivations()[i] = (float) (Math.pow(Math.E, parentLayerActivations[i]) / softmaxCoefficient);
        }
    }

    @Override
    public void backprop(float[] errors, float learningRate) {
        // No need to update weights and biases for this layer as there are none.

        IBasicLayer parentLayer = getParentLayer();

        float[] activations = getActivations();
        float[] parentError = new float[parentLayer.getSize()];

        for (int i = 0; i < getSize(); i++) {
            for (int j = 0; j < parentLayer.getSize(); j++) {
                // Cross-product of errors with Jacobian.
                parentError[j] += errors[i] * (i == j ? activations[i] * (1 - activations[i]) : -activations[i] * activations[j]);
            }
        }

        parentLayer.backprop(parentError, learningRate);
    }

    public static SoftmaxLayer build(int size) {
        return new SoftmaxLayer(Layer.build(size));
    }
}
