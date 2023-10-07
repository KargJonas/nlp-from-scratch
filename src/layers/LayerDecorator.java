package layers;

import util.Activations;
import util.Initializers;
import util.LayerType;

public class LayerDecorator implements ILayer {
    protected ILayer layer;

    LayerDecorator(ILayer layer) {
        this.layer = layer;
    }

    @Override
    public int getSize() {
        return layer.getSize();
    }

    @Override
    public float[] getActivations() {
        return layer.getActivations();
    }

    @Override
    public void setActivations(float[] activations) {
        layer.setActivations(activations);
    }

    @Override
    public IBasicLayer getParentLayer() {
        return layer.getParentLayer();
    }

    @Override
    public LayerType getLayerType() {
        return layer.getLayerType();
    }

    @Override
    public void setParentLayer(IBasicLayer parentLayer) {
        layer.setParentLayer(parentLayer);
    }

    @Override
    public ILayer setWeightInitializer(Initializers.Supplier weightInitializer) {
        layer.setWeightInitializer(weightInitializer);
        return this;
    }

    @Override
    public ILayer setBiasInitializer(Initializers.Supplier biasInitializer) {
        layer.setBiasInitializer(biasInitializer);
        return this;
    }

    @Override
    public ILayer setActivationInitializer(Initializers.Supplier activationInitializer) {
        layer.setActivationInitializer(activationInitializer);
        return this;
    }

    @Override
    public ILayer setActivationFn(Activations.ActivationFn activationFn) {
        layer.setActivationFn(activationFn);
        return this;
    }

    @Override
    public Activations.ActivationFn getActivationFn() {
        return layer.getActivationFn();
    }

    @Override
    public float[][] getWeights() {
        return layer.getWeights();
    }

    @Override
    public void setWeights(float[][] weights) {
        layer.setWeights(weights);
    }

    @Override
    public float[] getBiases() {
        return layer.getBiases();
    }

    @Override
    public void setBiases(float[] biases) {
        layer.setBiases(biases);
    }

    @Override
    public void initialize() {
        layer.initialize();
    }

    @Override
    public void computeActivations() { }

    @Override
    public void backprop(float[] errors, float learningRate) {
        layer.backprop(errors, learningRate);
    }
}
