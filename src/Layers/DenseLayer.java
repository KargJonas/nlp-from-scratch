package Layers;

import Util.LayerType;

public class DenseLayer extends Layer<DenseLayer> {
    DenseLayer(int size) {
        super(size);
        layerType  = LayerType.DENSE;
    }

    @Override
    protected DenseLayer getThis() {
        return this;
    }

    /**
     * Determine number of weights to previous layer,
     * initialize the arrays for the weights and biases.
     * !! initialize() does not set the values in the arrays !!
     * it just creates them.
     * initializeValues() is used to set the values
     */
    @Override
    public void initialize() {
        int parentLayerSize = parentLayer == null ? 0 : parentLayer.getSize();
        setWeightsPerUnit(getSize() * parentLayerSize);

        super.initialize();
    }

    public static DenseLayer build(int size) {
        return new DenseLayer(size);
    }

    /**
     * Computes the activations based on weights, biases and the previous layers' activations.
     */
    @Override
    public void computeActivations() {
        if (parentLayer == null) return;

        for (int i = 0; i < size; i++) {
            double weightedSum = 0;

            for (int j = 0; j < parentLayer.size; j++) {
                weightedSum += parentLayer.activations[j] * weights[i][j];
            }

            // TODO: This needs to be adjusted to work with functions such as SOFTMAX
            activations[i] = activationFn.f(weightedSum + biases[i]);
        }
    }
}
