import java.util.ArrayList;
import java.util.Arrays;

import layers.Layer;
import preprocessing.vectorization.Sample;
import util.LossFunctions.LossFn;
import util.LayerType;
import preprocessing.Preprocessor;

public class Model {
  ArrayList<Layer<?>> layers = new ArrayList<>();

  LossFn lossFunction;

  public Model addLayer(Layer<?> layer) {
    if (layers.size() == 0 && layer.layerType != LayerType.INPUT) {
      throw new RuntimeException("First layer must be of type InputLayer.");
    }

    if (layer.layerType == LayerType.SOFTMAX
      && getLastLayer().getSize() != layer.getSize()) {
      throw new RuntimeException("Softmax layer must be same shape as the layer before it.");
    }

    layer.setLayerIndex(layers.size());

    layers.add(layer);
    return this;
  }

  public Model setLossFunction(LossFn lossFunction) {
    this.lossFunction = lossFunction;
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
   * - Layer.setWeightsPerUnit()
   * - Layer.initialize()
   * - initializeValues()
   */
  public Model initialize() {
    System.out.println("Initializing model ...");

    if (layers.size() < 1) {
      System.out.println("\tNo layers in model. Aborting.");
      return this;
    }

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

    System.out.println("\tDone.");

    return this;
  }

  public void forwardPass(double[] input) {
    layers.get(0).setActivations(input);

    for (int i = 1; i < layers.size(); i++) {
      layers.get(i).computeActivations();
    }
  }

  /**
   * Returns the error between the activations of the last layer and the label.
   *
   * @param label Target output.
   * @return Error vector (output - label).
   */
  public double[] computeError(double[] label) {
    double[] prediction = getOutput();
    double[] error = new double[label.length];

    for (int i = 0; i < label.length; i++) {
      error[i] = prediction[i] - label[i];
    }

    return error;
  }

  /**
   * Applies the loss function to the label and the activations of the last layer.
   *
   * @param label Target output.
   * @return Loss value (scalar),
   */
  public double computeLoss(double[] label) {
    double[] output = getOutput();
    return lossFunction.f(output, label);
  }

  /**
   * @return Activations of the last layer.
   */
  public double[] getOutput() {
    return getLastLayer().getActivations();
  }

  public Layer<?> getLastLayer() {
    return layers.get(layers.size() - 1);
  }

  public Model train(
    Preprocessor preprocessor,
    int nEpochs,
    double learningRate
  ) {

    System.out.println("Training model ...");

    // Go through the entire training data nEpochs times
    for (int epochNumber = 0; epochNumber < nEpochs; epochNumber++) {

      double loss = 0;
      int batchNumber = 0;

      for (Sample[] batch : preprocessor) {
        double batchError = 0;

        for (Sample sample : batch) {

          // TODO: This just here for testing purposes and is a MAJOR bottleneck
          double[] data = Arrays.stream(sample.data())
              .flatMapToDouble(Arrays::stream)
                .toArray();

          forwardPass(data);
          batchError += computeLoss(sample.label());
          getLastLayer().backprop(computeError(sample.label()), learningRate);
        }

        batchError /= preprocessor.getBatchSize();
        loss = (loss * batchNumber + batchError) / (batchNumber + 1);

//        System.out.printf("Batch %s   Epoch %s/%s %.0f%%)   Batch error: %.8f\n",
//          batchNumber, epochNumber, nEpochs, (epochNumber + 1.) / nEpochs * 100., batchError);

        batchNumber++;
      }

      System.out.printf("Epoch %s/%s %.0f%%)   Loss: %.8f\n",
        epochNumber, nEpochs, (epochNumber + 1.) / nEpochs * 100., loss);

//      double[] input = sampleProvider.next();
//      double[] label = labelProvider.next();
//
//      int batchNumber = 0;
//
//      outer:
//      while (input != null && label != null) {
//        double batchError = 0;
//        int j;
//        batchNumber++;
//
//        for (j = 0; j < batchSize; j++) {
//          if (label == null || input == null) break outer;
//
//          forwardPass(input);
//          batchError += computeLoss(label);
//          getLastLayer().backprop(computeError(label), learningRate);
//
//          input = sampleProvider.next();
//          label = labelProvider.next();
//        }
//
//        batchError /= (j + 1);
//
//        System.out.printf(
//          "\tBatch %s, Epoch %s/%s = %s%%, Batch error: %s\n",
//          batchNumber,
//          epochNumber,
//          nEpochs,
//          Math.round((epochNumber / (double) nEpochs) * 10000d) / 100d,
//          batchError);
//      }
    }

    return this;
  }
}
