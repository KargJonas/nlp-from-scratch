package layers;

public class RecurrentLayer extends DenseLayer {

  RecurrentLayer(int size) {
    super(size);
  }

  @Override
  public void computeActivations() {
    if (parentLayer == null) {
      throw new RuntimeException("No parent layer");
    }

    if (parentLayer.getSize() != getSize()) {
      throw new RuntimeException("RecurrentLayer expects parent layer of same size.");
    }

    for (int i = 0; i < getSize(); i++) {
      float weightedSum = 0;

      for (int j = 0; j < parentLayer.getSize(); j++) {
        weightedSum += parentLayer.activations[j] * weights[i][j];
      }

      // TODO: here lies the magic: +=
      activations[i] = (activations[i] + activationFn.f(weightedSum + biases[i])) / 2f;
    }
  }

  public static RecurrentLayer build(int size) {
    return new RecurrentLayer(size);
  }
}
