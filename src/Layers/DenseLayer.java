package Layers;

import Util.Initializers;

public class DenseLayer extends Layer {
    DenseLayer(int size) {
        super(size);
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
        setWBSizes(getSize() * parentLayerSize, getSize());
        super.initialize();
    }

    public static DenseLayer build(int size) {
        return new DenseLayer(size);
    }

    public static DenseLayer build(int size, Initializers.Supplier weightInitializer) {
        return (DenseLayer)Layer.build(size, weightInitializer);
    }

    public static DenseLayer build(int size, Initializers.Supplier weightInitializer, Initializers.Supplier biasInitializer) {
        return (DenseLayer)Layer.build(size, weightInitializer, biasInitializer);
    }
}
