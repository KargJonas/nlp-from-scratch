package layers;

import util.Activations;
import util.Initializers;

public interface ILayer extends IBasicLayer {
    void setParentLayer(GenericLayer parentLayer);

    GenericLayer setWeightInitializer(Initializers.Supplier weightInitializer);

    GenericLayer setBiasInitializer(Initializers.Supplier biasInitializer);

    GenericLayer setActivationInitializer(Initializers.Supplier activationInitializer);

    GenericLayer setActivationFn(Activations.ActivationFn activationFn);

    void initialize();

    void computeActivations();

    void backprop(float[] errors, float learningRate);
}
