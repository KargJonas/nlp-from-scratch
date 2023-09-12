import layers.Layer;
import preprocessing.BatchedPreprocessor;
import preprocessing.vectorization.Sample;
import util.LayerType;
import util.LossFunctions.LossFn;

import java.util.ArrayList;
import java.util.Arrays;

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

    BatchedPreprocessor preprocessor,
    int nEpochs,
    double learningRate
  ) {
    System.out.println("Training model ...");
    System.out.printf("\tNumber of epochs: %s\tBatch size: %s\n\n", nEpochs, preprocessor.getBatchSize());

    for (int i = 0; i < nEpochs; i++) {
      for (Sample[] batch : preprocessor) {

        double meanError = 0;

        for (Sample sample : batch) {

          // TODO: This just here for testing purposes and is a MAJOR bottleneck
          double[] input = Arrays.stream(sample.data())
            .flatMapToDouble(Arrays::stream)
            .toArray();

          double[] label = sample.label();

          forwardPass(input);
          double x = computeLoss(label);
          meanError += x;
          getLastLayer().backprop(computeError(label), learningRate);

//            System.out.println(x + "  " + preprocessor.decode(sample) + " Prediction: " + preprocessor.decode(getOutput()));

//            if (meanError / preprocessor.getBatchSize() > 0.2) {
//              System.out.printf("%s (Prediction was: \"%s\")\n", preprocessor.decode(sample), preprocessor.decode(getOutput()));
//            }
        }

        meanError /= preprocessor.getBatchSize();

        float percentage = ((float) i / nEpochs) * 100;
        System.out.printf("Epoch: %s/%s (%.0f%%)   Batch loss: %s\n", i, nEpochs, percentage, meanError);
      }
    }

    return this;
  }
}
