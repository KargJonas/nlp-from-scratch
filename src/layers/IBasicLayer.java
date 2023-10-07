package layers;

import util.LayerType;

import java.io.Serializable;

/**
 * IBasicLayer represents the set of methods that are necessary
 */
public interface IBasicLayer extends Serializable {

    int getSize();

    float[] getActivations();

    void setActivations(float[] activations);

    void setParentLayer(IBasicLayer parentLayer);

    IBasicLayer getParentLayer();

    void backprop(float[] errors, float learningRate);

    void initialize();

    void computeActivations();

    LayerType getLayerType();
}
