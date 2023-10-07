package layers;

import util.LayerType;

class BasicLayer implements IBasicLayer {

    // The activations are marked as transient to exclude them from serialization.
    // Activations are intermediate values that are used during training/inference
    // so there is no need to store them in the model checkpoints.
    transient protected float[] activations;

    // Number of units in the layer
    protected final int size;

    // The parent layer is the neighboring layer that is closest to the model input.
    // This layer consumes the output data of the parent layer as inputs.
    protected IBasicLayer parentLayer;

    protected BasicLayer(int size) {
        this.size = size;
    }

    @Override
    public void initialize() {
        // No need to initialize the activations
        activations = new float[getSize()];
    }

    @Override
    public void computeActivations() { }

    @Override
    public LayerType getLayerType() {
        return LayerType.BASIC;
    }

    @Override
    public int getSize() {
        return size;
    }

    @Override
    public float[] getActivations() {
        return activations;
    }

    @Override
    public void setActivations(float[] activations) {
        this.activations = activations;
    }

    @Override
    public void setParentLayer(IBasicLayer parentLayer) {
        this.parentLayer = parentLayer;
    }

    @Override
    public IBasicLayer getParentLayer() {
        return parentLayer;
    }

    @Override
    public void backprop(float[] errors, float learningRate) { }

    public static IBasicLayer build(int size) {
        return new BasicLayer(size);
    }
}
