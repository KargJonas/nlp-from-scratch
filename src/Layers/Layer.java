package Layers;

import Util.Activation;
import Util.Initializers;
import Util.LayerType;

public abstract class Layer<T extends Layer<T>> {
    public LayerType layerType = LayerType.ABSTRACT;
    protected final int size;
    protected double[][] weights;
    protected double[] biases;
    protected double[] activations;
    private Integer weightsPerUnit;
    protected Layer<?> parentLayer;
    protected Activation.ActivationFn activationFn = Activation.IDENTITY;

    private Initializers.Supplier weightInitializer;
    private Initializers.Supplier biasInitializer = Initializers.ZEROS();
    private Initializers.Supplier activationInitializer = Initializers.GAUSSIAN(0, 0.01);

    Layer(int size) {
        this.size = size;
        weightInitializer = Initializers.GLOROT(size);
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

    public T setActivationInitializer(Initializers.Supplier activationInitializer) {
        this.activationInitializer = activationInitializer;
        return getThis();
    }

    public T setActivationFn(Activation.ActivationFn activationFn) {
        this.activationFn = activationFn;
        return getThis();
    }

    /**
     * Set size of the weights array.
     * This number depends on the size of the parent layer and the type of layer.
     * @param weightsPerUnit Size of the weights array.
     */
    protected void setWeightsPerUnit(int weightsPerUnit) {
        this.weightsPerUnit = weightsPerUnit;
    }

    /**
     * Allocates memory for weights[], biases[] and activations[].
     */
    public void initialize() {
        if (weightsPerUnit == null) {
            throw new RuntimeException("Layer.initialize() called before setWeightsPerUnit()");
        }

        weights = new double[size][weightsPerUnit];
        biases = new double[size];
        activations = new double[size];
    }

    /**
     * Initializes the values of weights[], biases[] and activations[] using their respective initializers.
     * This method may only be called when weights[], biases[] and activations[] have been initialized.
     */
    public void initializeValues() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < weightsPerUnit; j++) {
                weights[i][j] = weightInitializer.get();
            }
        }

        for (int i = 0; i < size; i++) {
            biases[i] = biasInitializer.get();
        }

        for (int i = 0; i < size; i++) {
            biases[i] = activationInitializer.get();
        }
    }

    public void setActivations(double[] activations) {
        this.activations = activations;
    }

    /**
     * Do not remove, this is here so that there is always a callable computeActivations available,
     * even for layers where this does not make sense and is thus not implemented, e.g. input layer.
     */
    public void computeActivations() { }

    public int getSize() {
        return size;
    }

    public double[] getActivations() {
        return activations;
    }
}
