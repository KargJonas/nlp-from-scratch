package layers;

import util.LayerType;

public class InputLayer extends BasicLayer {
    protected InputLayer(int size) {
        super(size);
    }

    @Override
    public LayerType getLayerType() {
        return LayerType.INPUT;
    }

    public static IBasicLayer build(int size) {
        return new InputLayer(size);
    }
}
