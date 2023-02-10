package Layers;

public class DenseLayer extends Layer<DenseLayer> {
    DenseLayer(int size) {
        super(size);
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
        setWBSizes(getSize() * parentLayerSize, getSize());
        super.initialize();
    }

    public static DenseLayer build(int size) {
        return new DenseLayer(size);
    }
}
