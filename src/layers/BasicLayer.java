package layers;

import util.LayerType;

public class BasicLayer implements IBasicLayer {

    // The activations are marked as transient to exclude them from serialization.
    // Activations are intermediate values that are used during training/inference
    // so there is no need to store them in the model checkpoints.
    transient protected float[] activations;

    // Number of units in the layer
    protected final int size;

    protected BasicLayer(int size) {
        this.size = size;
    }

    @Override
    public void initialize() {
        // No need to initialize the activations
        activations = new float[getSize()];
    }

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
    public void backprop(float[] errors, float learningRate) { }

    public static IBasicLayer build(int size) {
        return new BasicLayer(size);
    }
}
