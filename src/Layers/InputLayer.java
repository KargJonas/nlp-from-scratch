package Layers;

import Util.Shape;

public class InputLayer extends Layer<InputLayer> {
    Shape shape;

    public InputLayer(Shape shape) {
        super(shape.getSize());
        this.shape = shape;
    }

    @Override
    protected InputLayer getThis() {
        return this;
    }

    @Override
    public void initialize() {
        setWBSizes(0, 0);
        super.initialize();
    }

    public static InputLayer build(Shape shape) {
        return new InputLayer(shape);
    }
}
