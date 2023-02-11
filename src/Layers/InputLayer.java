package Layers;

import Util.LayerType;
import Util.Shape;

public class InputLayer extends Layer<InputLayer> {
    Shape shape;

    public InputLayer(Shape shape) {
        super(shape.getSize());
        layerType = LayerType.INPUT;
        this.shape = shape;
    }

    @Override
    protected InputLayer getThis() {
        return this;
    }

    @Override
    public void initialize() {
        // No need to initialize
        activations = new double[size];
    }

    @Override
    public void initializeValues() { }

    public static InputLayer build(Shape shape) {
        return new InputLayer(shape);
    }
}
