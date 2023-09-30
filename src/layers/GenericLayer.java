package layers;

import util.Activations;
import util.Initializers;
import util.LayerType;

import java.io.Serializable;

public abstract class GenericLayer<T extends GenericLayer<T>> implements Serializable {
    public LayerType layerType = LayerType.ABSTRACT;
    private int layerIndex;
    private final int size;
    protected float[][] weights;
    protected float[] biases;
    transient protected float[] activations;
    private Integer weightsPerUnit;
    protected GenericLayer<?> parentLayer;
    protected Activations.ActivationFn activationFn = Activations.IDENTITY;

    private Initializers.Supplier weightInitializer;
    private Initializers.Supplier biasInitializer = Initializers.ZEROS();
    private Initializers.Supplier activationInitializer = Initializers.GAUSSIAN(0, 0.01f);

    GenericLayer(int size) {
        this.size = size;
        weightInitializer = Initializers.GLOROT(size);
    }

    protected abstract T getThis();

    public void setParentLayer(GenericLayer<?> parentLayer) {
        this.parentLayer = parentLayer;
    }

    public void setLayerIndex(int layerIndex) {
        this.layerIndex = layerIndex;
    }

    public int getLayerIndex() {
        return layerIndex;
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

    public T setActivationFn(Activations.ActivationFn activationFn) {
        this.activationFn = activationFn;
        return getThis();
    }

    /**
     * Set size of the weights array.
     * This number depends on the size of the parent layer and the type of layer.
     *
     * @param weightsPerUnit Size of the weights array.
     */
    protected void setWeightsPerUnit(int weightsPerUnit) {
        this.weightsPerUnit = weightsPerUnit;
    }

    /**
     * Allocates memory for weights[], biases[] and activations[] and initializes their
     * values using their respective initializers if necessary.
     */
    public void initialize() {
        if (weightsPerUnit == null) {
            throw new RuntimeException("Layer.initialize() called before setWeightsPerUnit()");
        }

        if (weights == null) {
            weights = new float[size][weightsPerUnit];

            for (int i = 0; i < size; i++) {
                for (int j = 0; j < weightsPerUnit; j++) {
                    weights[i][j] = weightInitializer.get();
                }
            }
        }

        if (biases == null) {
            biases = new float[size];

            for (int i = 0; i < size; i++) {
                biases[i] = biasInitializer.get();
            }
        }

        if (activations == null) {
            activations = new float[size];

            for (int i = 0; i < size; i++) {
                activations[i] = activationInitializer.get();
            }
        }
    }

    public void setActivations(float[] activations) {
        this.activations = activations;
    }

    /**
     * Do not remove, this is here so that there is always a callable computeActivations available,
     * even for layers where this does not make sense and is thus not implemented, e.g. input layer.
     */
    public void computeActivations() {
    }

    public int getSize() {
        return size;
    }

    public float[] getActivations() {
        return activations;
    }

    public void backprop(float[] errors, float learningRate) {
        float[] newBiases = new float[size];
        float[][] newWeights = new float[size][weightsPerUnit];
        float[] parentActivations = parentLayer.getActivations();

        for (int i = 0; i < size; i++) {
            // Compute delta for the output layer
            float delta = errors[i] * activationFn.df(activations[i]);

            // Update biases
            newBiases[i] = biases[i] - learningRate * delta;

            // Update weights
            for (int j = 0; j < weightsPerUnit; j++) {
                newWeights[i][j] = weights[i][j] - learningRate * delta * parentActivations[j];
            }
        }

        biases = newBiases;
        weights = newWeights;

        // Compute delta for the hidden layer
        float[] parentErrors = new float[parentLayer.getSize()];

        for (int j = 0; j < weightsPerUnit; j++) {
            for (int i = 0; i < size; i++) {
                parentErrors[j] += errors[i] * weights[i][j] * activationFn.df(activations[i]);
            }
        }

        parentLayer.backprop(parentErrors, learningRate);
    }
}
