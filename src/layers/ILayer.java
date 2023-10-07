package layers;

import util.Activations;
import util.Initializers;

public interface ILayer extends IBasicLayer {
    ILayer setWeightInitializer(Initializers.Supplier weightInitializer);

    ILayer setBiasInitializer(Initializers.Supplier biasInitializer);

    ILayer setActivationInitializer(Initializers.Supplier activationInitializer);

    ILayer setActivationFn(Activations.ActivationFn activationFn);

    Activations.ActivationFn getActivationFn();

    float[][] getWeights();

    void setWeights(float[][] weights);

    float[] getBiases();

    void setBiases(float[] biases);

    void initialize();

    void backprop(float[] errors, float learningRate);
}
