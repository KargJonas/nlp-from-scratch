package layers;

import util.LayerType;
import util.Shape;

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
        activations = new float[getSize()];
    }

    public static InputLayer build(Shape shape) {
        return new InputLayer(shape);
    }

    @Override
    public void backprop(float[] errors, float learningRate) { }
}
