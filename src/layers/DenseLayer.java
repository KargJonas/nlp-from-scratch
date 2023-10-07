package layers;

public class DenseLayer extends LayerDecorator {
    protected DenseLayer(ILayer baseLayer) {
        super(baseLayer);
    }

    /**
     * Computes the activations based on weights, biases and the previous layers' activations.
     */
    @Override
    public void computeActivations() {
        IBasicLayer parentLayer = getParentLayer();
        if (parentLayer == null) return;

        for (int i = 0; i < getSize(); i++) {
            float weightedSum = 0;

            for (int j = 0; j < parentLayer.getSize(); j++) {
                weightedSum += parentLayer.getActivations()[j] * getWeights()[i][j];
            }

            getActivations()[i] = getActivationFn().f(weightedSum + getBiases()[i]);
        }
    }

    public static DenseLayer build(int size) {
        return new DenseLayer(Layer.build(size));
    }
}
