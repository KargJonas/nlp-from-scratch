package layers;

import util.LayerType;

public class DenseLayer extends GenericLayer<DenseLayer> {
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
//        setWeightsPerUnit(parentLayerSize);

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

        for (int i = 0; i < getSize(); i++) {
            float weightedSum = 0;

            for (int j = 0; j < parentLayer.getSize(); j++) {
                weightedSum += parentLayer.getActivations()[j] * weights[i][j];
            }

            activations[i] = activationFn.f(weightedSum + biases[i]);
        }
    }
}
