package Layers;

import Util.Initializers;

public class Layer {
    private final int size;

    double[] weights;
    double[] biases;
    Integer weightSize;
    Integer biasSize;

    Layer parentLayer;

    Initializers.Supplier weightInitializer = Initializers.RAND(0.1);
    Initializers.Supplier biasInitializer = Initializers.ZEROS();

    Layer(int size) {
        this.size = size;
    }

    public void setParentLayer(Layer parentLayer) {
        this.parentLayer = parentLayer;
    }

    public void setWeightInitializer(Initializers.Supplier weightInitializer) {
        this.weightInitializer = weightInitializer;
    }

    public void setBiasInitializer(Initializers.Supplier biasInitializer) {
        this.biasInitializer = biasInitializer;
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

    protected static Layer build(int size) {
        return new Layer(size);
    }

    protected static Layer build(int size, Initializers.Supplier weightInitializer) {
        Layer builtLayer = new Layer(size);
        builtLayer.setWeightInitializer(weightInitializer);
        return builtLayer;
    }

    protected static Layer build(int size, Initializers.Supplier weightInitializer, Initializers.Supplier biasInitializer) {
        Layer builtLayer = Layer.build(size, weightInitializer);
        builtLayer.setBiasInitializer(biasInitializer);
        return builtLayer;
    }
}
