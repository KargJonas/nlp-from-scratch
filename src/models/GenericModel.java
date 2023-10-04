package models;

import checkpoint.CheckpointManager;
import layers.GenericLayer;
import preprocessing.TrainingDataPreprocessor;
import preprocessing.vectorization.Sample;
import telemetry.TrainingMonitor;
import util.LayerType;
import util.LossFunctions.LossFn;

import java.io.Serializable;
import java.util.ArrayList;

public class GenericModel implements Model, Serializable {

  public String name = "unnamed-model";
  transient TrainingMonitor trainingMonitor;
  transient CheckpointManager checkpointManager;
  ArrayList<GenericLayer<?>> layers = new ArrayList<>();
  LossFn lossFunction;
  long checkpointNumber = 0;

  @Override
  public CheckpointManager getCheckpointManager() {
    return checkpointManager;
  }

  public GenericModel addLayer(GenericLayer<?> layer) {
    if (layers.isEmpty() && layer.layerType != LayerType.INPUT) {
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

  public Model setName(String name) {
    this.name = name;
    return this;
  }

  public Model setLossFunction(LossFn lossFunction) {
    this.lossFunction = lossFunction;
    return this;
  }

  public Model attachTelemetry(TrainingMonitor trainingMonitor) {
    trainingMonitor.setName(getName());
    this.trainingMonitor = trainingMonitor;
    return this;
  }

  public Model attachCheckpointManager(CheckpointManager checkpointManager) {
    checkpointManager.setName(getName());
    this.checkpointManager = checkpointManager;
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

    if (layers.isEmpty()) {
      throw new RuntimeException("No layers in model. Aborting.");
    }

    // Set parent layers
    for (int i = 1; i < layers.size(); i++) {
      layers.get(i).setParentLayer(layers.get(i - 1));
    }

    for (GenericLayer<?> layer : layers) {
      // Create arrays for weights and biases of appropriate size.
      layer.initialize(); // !! this must call initialize for

      // Initialize the values in the arrays using the provided/default Initializers.
//      layer.initializeValues();
    }

    System.out.println("\tDone.");

    return this;
  }

  public void forwardPass(float[] input) {
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
  public float[] computeError(float[] label) {
    float[] prediction = getOutput();
    float[] error = new float[label.length];

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
  public float computeLoss(float[] label) {
    float[] output = getOutput();
    return lossFunction.f(output, label);
  }

  /**
   * @return Activations of the last layer.
   */
  public float[] getOutput() {
    return getLastLayer().getActivations();
  }

  /**
   * Returns the last/output layer of the network.
   * @return The output layer of the network.
   */
  GenericLayer<?> getLastLayer() {
    return layers.get(layers.size() - 1);
  }

  public Model train(
    TrainingDataPreprocessor preprocessor,
    int nEpochs,
    float learningRate
  ) {
    try {
      System.out.println("Training model ...");
      System.out.printf("\tNumber of epochs: %s\tBatch size: %s\n\n", nEpochs, preprocessor.getBatchSize());

      long totalBatches = nEpochs * preprocessor.getBatchCount();
      int batchNumber = 0;

      for (int i = 0; i < nEpochs; i++) {

        for (Sample[] batch : preprocessor) {

          float meanError = 0;

          for (Sample sample : batch) {

            // TODO: This just here for testing purposes and is a MAJOR bottleneck
//            float[] input = Model.floatFlat(sample.data());
            float[] input = sample.data();
            float[] label = sample.label();

            forwardPass(input);
            float x = computeLoss(label);
            meanError += x;
            getLastLayer().backprop(computeError(label), learningRate);
            checkpointNumber++;
          }

          meanError /= preprocessor.getBatchSize();

          if (trainingMonitor != null) {
            trainingMonitor.add(meanError);
          }

          float percentage = ((float) batchNumber / totalBatches) * 100;
          System.out.printf("Epoch: %s/%s (%.0f%%)   Batch loss: %s\n", i, nEpochs, percentage, meanError);

          batchNumber++;
        }
      }
    } catch (Exception e) {
      System.out.println("Training failed:");
      throw e;
    }

    return this;
  }

  @Override
  public String getName() {
    return name;
  }

  public Model commitMetrics() {
    if (trainingMonitor == null) throw new RuntimeException("Cant commit training metrics: No training monitor configured.");
    trainingMonitor.commit();
    return this;
  }

  public Model createCheckpoint() {
    if (checkpointManager == null) throw new RuntimeException("Cant create checkpoint: No checkpoint manager configured.");
    checkpointManager.createCheckpoint(this);
    return this;
  }
}