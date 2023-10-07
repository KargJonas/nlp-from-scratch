package layers;

import util.Activations;
import util.Initializers;
import util.LayerType;

/**
 * Provides the necessary functionality to construct, configure and initialize a generic layer.
 * A generic layer has weights, biases, activations, a parent layer, an activation function and initializers.
 */
public class GenericLayer extends BasicLayer implements ILayer {

    protected float[][] weights;    // The weights connect this layer to the parent layer.
    protected float[] biases;       // The biases act as "offsets" of the weighted sums.

    // Weights per unit is the number of weights that each unit is attached with to the parent layer.
    // weightsPerUnit is always equal to the number of units in the parent layer.
    protected Integer weightsPerUnit;

    // The parent layer is the neighboring layer that is closest to the model input.
    // This layer consumes the output data of the parent layer as inputs.
    protected IBasicLayer parentLayer;

    // Default value for activation function is IDENTITY
    protected Activations.ActivationFn activationFn = Activations.IDENTITY;

    // Initializers provide initial values of the weights/biases/activations
    private Initializers.Supplier weightInitializer;
    private Initializers.Supplier biasInitializer;
    private Initializers.Supplier activationInitializer;

    GenericLayer(int size) {
        super(size);

        // Configuring the initializer defaults
        weightInitializer = Initializers.GLOROT(size);
        biasInitializer = Initializers.ZEROS();
        activationInitializer = Initializers.GAUSSIAN(0, 0.01f);
    }

    @Override
    public LayerType getLayerType() {
        return LayerType.GENERIC;
    }

    @Override
    public void setParentLayer(GenericLayer parentLayer) {
        this.parentLayer = parentLayer;
        this.weightsPerUnit = parentLayer.getSize();
    }

    @Override
    public GenericLayer setWeightInitializer(Initializers.Supplier weightInitializer) {
        this.weightInitializer = weightInitializer;
        return this;
    }

    @Override
    public GenericLayer setBiasInitializer(Initializers.Supplier biasInitializer) {
        this.biasInitializer = biasInitializer;
        return this;
    }

    @Override
    public GenericLayer setActivationInitializer(Initializers.Supplier activationInitializer) {
        this.activationInitializer = activationInitializer;
        return this;
    }

    @Override
    public GenericLayer setActivationFn(Activations.ActivationFn activationFn) {
        this.activationFn = activationFn;
        return this;
    }

    /**
     * Allocates memory for weights[], biases[] and activations[] and initializes their
     * values using their respective initializers if necessary.
     */
    @Override
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

    /**
     * Do not remove, this is here so that there is always a callable computeActivations available,
     * even for layers where this does not make sense and is thus not implemented, e.g. input layer.
     */
    @Override
    public void computeActivations() { }

    @Override
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

    public static ILayer build(int size) {
        return new GenericLayer(size);
    }
}
