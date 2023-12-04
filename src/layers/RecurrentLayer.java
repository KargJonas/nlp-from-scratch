////package layers;
////
////public class RecurrentLayer extends DenseLayer {
////
////  RecurrentLayer(int size) {
////    super(size);
////  }
////
////  @Override
////  public void computeActivations() {
////    if (parentLayer == null) {
////      throw new RuntimeException("No parent layer");
////    }
////
////    if (parentLayer.getSize() != getSize()) {
////      throw new RuntimeException("RecurrentLayer expects parent layer of same size.");
////    }
////
////    for (int i = 0; i < getSize(); i++) {
////      float weightedSum = 0;
////
////      for (int j = 0; j < parentLayer.getSize(); j++) {
////        weightedSum += parentLayer.activations[j] * weights[i][j];
////      }
////
////      // TODO: here lies the magic: +=
////      activations[i] = (activations[i] + activationFn.f(weightedSum + biases[i])) / 2f;
////    }
////  }
////
////  public static RecurrentLayer build(int size) {
////    return new RecurrentLayer(size);
////  }
////}
//
//
//package layers;
//
//import util.Initializers;
//
//public class RecurrentLayer extends LayerDecorator {
//  private final float[] hiddenActivations;
//
//  RecurrentLayer(ILayer layer) {
//    super(layer);
//    hiddenActivations = new float[getSize()];
//  }
//
//  public void initialize() {
//    super.initialize();
//
//    for (int i = 0; i < getSize(); i++) {
//      hiddenActivations[i] = Initializers.GLOROT(getSize()).get();
//    }
//  }
//
//  @Override
//  public void computeActivations() {
//    if (getParentLayer() == null) {
//      throw new RuntimeException("No parent layer");
//    }
//
//    if (getParentLayer().getSize() != getSize()) {
//      throw new RuntimeException("RecurrentLayer expects parent layer of the same size.");
//    }
//
//    for (int i = 0; i < getSize(); i++) {
//      float weightedSum = 0;
//
//      for (int j = 0; j < getParentLayer().getSize(); j++) {
//        weightedSum += getParentLayer().getActivations()[j] * getWeights()[i][j];
//      }
//
//      // Update the activations using the previous activations
//      getActivations()[i] = getActivationFn().f(weightedSum + getBiases()[i] + hiddenActivations[i]);
//      // Store the current activations as previous activations for the next time step
//      hiddenActivations[i] = getActivations()[i];
//    }
//  }
//
//  @Override
//  public void backprop(float[] errors, float learningRate) {
//    IBasicLayer parentLayer = getParentLayer();
//    int weightsPerUnit = parentLayer.getSize();
//    int size = getSize();
//
//    float[] newBiases = new float[size];
//    float[][] newWeights = new float[size][weightsPerUnit];
//    float[] parentActivations = parentLayer.getActivations();
//
//    for (int i = 0; i < size; i++) {
//      // Compute delta for the output layer
//      float delta = errors[i] * getActivationFn().df(getActivations()[i]);
//
//      // Update biases
//      newBiases[i] = getBiases()[i] - learningRate * delta;
//
//      // Update weights
//      for (int j = 0; j < weightsPerUnit; j++) {
//        newWeights[i][j] = getWeights()[i][j] - learningRate * delta * parentActivations[j];
//      }
//
//      // Compute delta for the hidden layer (including recurrent connections)
//      float recurrentDelta = delta * getActivationFn().df(getActivations()[i]);
//
//      for (int j = 0; j < size; j++) {
//        recurrentDelta += errors[j] * getWeights()[j][i] * getActivationFn().df(getActivations()[i]);
//      }
//
//      // Update biases for recurrent connections
//      newBiases[i] -= learningRate * recurrentDelta;
//    }
//
//    setWeights(newWeights);
//    setBiases(newBiases);
//
//    // Compute delta for the parent layer
//    float[] parentErrors = new float[parentLayer.getSize()];
//
//    for (int j = 0; j < weightsPerUnit; j++) {
//      for (int i = 0; i < size; i++) {
//        parentErrors[j] += errors[i] * getWeights()[i][j] * getActivationFn().df(getActivations()[i]);
//      }
//    }
//
//    parentLayer.backprop(parentErrors, learningRate);
//  }
//
//  public static RecurrentLayer build(int size) {
//    return new RecurrentLayer(Layer.build(size));
//  }
//}


package layers;

import util.Initializers;

public class RecurrentLayer extends LayerDecorator {
  private final float[] hiddenActivations;

  RecurrentLayer(ILayer layer) {
    super(layer);
    hiddenActivations = new float[getSize()];
  }

  public void initialize() {
    super.initialize();

    for (int i = 0; i < getSize(); i++) {
      hiddenActivations[i] = Initializers.GLOROT(getSize()).get();
    }
  }

  @Override
  public void computeActivations() {
    if (getParentLayer() == null) {
      throw new RuntimeException("No parent layer");
    }

//    if (getParentLayer().getSize() != getSize()) {
//      throw new RuntimeException("RecurrentLayer expects parent layer of the same size.");
//    }

    float[] logits = new float[getSize()]; // Store the raw logits

    for (int i = 0; i < getSize(); i++) {
      float weightedSum = 0;

      for (int j = 0; j < getParentLayer().getSize(); j++) {
        weightedSum += getParentLayer().getActivations()[j] * getWeights()[i][j];
      }

      // Update the activations using the previous activations
      logits[i] = weightedSum + getBiases()[i] + hiddenActivations[i];
      // Store the current activations as previous activations for the next time step
      hiddenActivations[i] = logits[i];
    }

    // Apply the softmax activation function to compute probabilities
    float[] activations = softmax(logits);
    setActivations(activations);
  }

  // Softmax activation function
  private float[] softmax(float[] logits) {
    float[] probabilities = new float[logits.length];
    float maxLogit = Float.NEGATIVE_INFINITY;

    // Find the maximum logit to prevent overflow
    for (float logit : logits) {
      if (logit > maxLogit) {
        maxLogit = logit;
      }
    }

    float sumExp = 0;

    // Compute the sum of exponentials for normalization
    for (int i = 0; i < logits.length; i++) {
      sumExp += Math.exp(logits[i] - maxLogit);
    }

    // Calculate softmax probabilities
    for (int i = 0; i < logits.length; i++) {
      probabilities[i] = (float) (Math.exp(logits[i] - maxLogit) / sumExp);
    }

    return probabilities;
  }

  @Override
  public void backprop(float[] errors, float learningRate) {
    // Implement backpropagation for softmax is a bit more complex,
    // you'll need to calculate gradients differently.
    // It's commonly done in the context of cross-entropy loss
    // which is typically used with softmax. If you need assistance
    // with that part, please let me know.

    // ... (backpropagation code for softmax)
  }

  public static RecurrentLayer build(int size) {
    return new RecurrentLayer(Layer.build(size));
  }
}
