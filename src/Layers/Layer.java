package Layers;

import Util.Activation;
import Util.Initializers;

public abstract class Layer<T extends Layer<T>> {
    private final int size;
    double[] weights;
    double[] biases;
    Integer weightSize;
    Integer biasSize;
    Layer<?> parentLayer;
    Activation.ActivationFn activationFn = Activation.IDENTITY;

    Initializers.Supplier weightInitializer = Initializers.RAND(0.1);
    Initializers.Supplier biasInitializer = Initializers.ZEROS();

    Layer(int size) {
        this.size = size;
    }

    protected abstract T getThis();

    public void setParentLayer(Layer<?> parentLayer) {
        this.parentLayer = parentLayer;
    }

    public T setWeightInitializer(Initializers.Supplier weightInitializer) {
        this.weightInitializer = weightInitializer;
        return getThis();
    }

    public T setBiasInitializer(Initializers.Supplier biasInitializer) {
        this.biasInitializer = biasInitializer;
        return getThis();
    }

    public T setActivationFn(Activation.ActivationFn activationFn) {
        this.activationFn = activationFn;
        return getThis();
    }

    protected void setWBSizes(int weightSize, int biasSize) {
        this.weightSize = weightSize;
        this.biasSize = biasSize;
    }

    // !! This should never be directly called. Only through inheriting classes. !!
    public void initialize() {
        if (weightSize == null || biasSize == null) {
            throw new RuntimeException("Layer.initialize() called before setWBSizes()");
        }

        weights = new double[weightSize];
        biases = new double[biasSize];
    }

    /**
     * This method may only be called when weights[] and biases[] have been initialized.
     */
    public void initializeValues() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = weightInitializer.get();
        }

        for (int i = 0; i < biases.length; i++) {
            biases[i] = biasInitializer.get();
        }
    }

    public int getSize() {
        return size;
    }
}
