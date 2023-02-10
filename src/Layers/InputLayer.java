package Layers;

import Util.Shape;

public class InputLayer extends Layer {
    Shape shape;

    public InputLayer(Shape shape) {
        super(shape.getSize());
        this.shape = shape;
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
