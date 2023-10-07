//package layers;
//
//public class RecurrentLayer extends DenseLayer {
//
//  RecurrentLayer(int size) {
//    super(size);
//  }
//
//  @Override
//  public void computeActivations() {
//    if (parentLayer == null) {
//      throw new RuntimeException("No parent layer");
//    }
//
//    if (parentLayer.getSize() != getSize()) {
//      throw new RuntimeException("RecurrentLayer expects parent layer of same size.");
//    }
//
//    for (int i = 0; i < getSize(); i++) {
//      float weightedSum = 0;
//
//      for (int j = 0; j < parentLayer.getSize(); j++) {
//        weightedSum += parentLayer.activations[j] * weights[i][j];
//      }
//
//      // TODO: here lies the magic: +=
//      activations[i] = (activations[i] + activationFn.f(weightedSum + biases[i])) / 2f;
//    }
//  }
//
//  public static RecurrentLayer build(int size) {
//    return new RecurrentLayer(size);
//  }
//}


package layers;

public class RecurrentLayer extends DenseLayer {
  private final float[] previousActivations;

  RecurrentLayer(int size) {
    super(size);
    previousActivations = new float[size];
  }

  @Override
  public void computeActivations() {
    if (parentLayer == null) {
      throw new RuntimeException("No parent layer");
    }

    if (parentLayer.getSize() != getSize()) {
      throw new RuntimeException("RecurrentLayer expects parent layer of the same size.");
    }

    for (int i = 0; i < getSize(); i++) {
      float weightedSum = 0;

      for (int j = 0; j < parentLayer.getSize(); j++) {
        weightedSum += parentLayer.activations[j] * weights[i][j];
      }

      // Update the activations using the previous activations
      activations[i] = activationFn.f(weightedSum + biases[i] + previousActivations[i]);
      // Store the current activations as previous activations for the next time step
      previousActivations[i] = activations[i];
    }
  }

  @Override
  public void backprop(float[] errors, float learningRate) {
    float[] newBiases = new float[getSize()];
    float[][] newWeights = new float[getSize()][weightsPerUnit];
    float[] parentActivations = parentLayer.getActivations();

    for (int i = 0; i < getSize(); i++) {
      // Compute delta for the output layer
      float delta = errors[i] * activationFn.df(activations[i]);

      // Update biases
      newBiases[i] = biases[i] - learningRate * delta;

      // Update weights
      for (int j = 0; j < weightsPerUnit; j++) {
        newWeights[i][j] = weights[i][j] - learningRate * delta * parentActivations[j];
      }

      // Compute delta for the hidden layer (including recurrent connections)
      float recurrentDelta = delta * activationFn.df(activations[i]);

      for (int j = 0; j < getSize(); j++) {
        recurrentDelta += errors[j] * weights[j][i] * activationFn.df(activations[i]);
      }

      // Update biases for recurrent connections
      newBiases[i] -= learningRate * recurrentDelta;
    }

    biases = newBiases;
    weights = newWeights;

    // Compute delta for the parent layer
    float[] parentErrors = new float[parentLayer.getSize()];

    for (int j = 0; j < weightsPerUnit; j++) {
      for (int i = 0; i < getSize(); i++) {
        parentErrors[j] += errors[i] * weights[i][j] * activationFn.df(activations[i]);
      }
    }

    parentLayer.backprop(parentErrors, learningRate);
  }


  public static RecurrentLayer build(int size) {
    return new RecurrentLayer(size);
  }
}
