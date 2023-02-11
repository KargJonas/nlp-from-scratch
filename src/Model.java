import java.util.ArrayList;

import Layers.Layer;

public class Model {
    ArrayList<Layer<?>> layers = new ArrayList<>();

    public Model addLayer(Layer<?> layer) {
        layers.add(layer);
        return this;
    }

    /**
     * Tells the layers who their parents are,
     * allocates memory for the weights and biases,
     * initializes the weights and biases using the provided initializers.
     * //
     * Call Chain should look like this:
     * - setParentLayer()
     * - Dense/Input/RNN.initialize()
     *      - Layer.setWBSizes()
     *      - Layer.initialize()
     * - initializeValues()
     */
    public Model initialize() {
        if (layers.size() < 1) return this;

        // Set parent layers
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).setParentLayer(layers.get(i - 1));
        }

        for (Layer<?> layer : layers) {
            // Create arrays for weights and biases of appropriate size.
            layer.initialize(); // !! this must call initialize for

            // Initialize the values in the arrays using the provided/default Initializers.
            layer.initializeValues();
        }

        return this;
    }
}
