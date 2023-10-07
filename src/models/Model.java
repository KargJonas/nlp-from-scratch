package models;

import checkpoint.CheckpointManager;
import layers.GenericLayer;
import preprocessing.TrainingDataPreprocessor;
import telemetry.TrainingMonitor;
import util.LossFunctions;

public interface Model {

  Model addLayer(GenericLayer<?> layer);

  CheckpointManager getCheckpointManager();

  Model setName(String name);

  Model setLossFunction(LossFunctions.LossFn lossFunction);

  /**
   * Attaches a training monitor for collecting metrics during training.
   * @param trainingMonitor The training monitor
   */
  Model attachTelemetry(TrainingMonitor trainingMonitor);

  /**
   * Attaches a checkpointManager for creating "snapshots" of the model.
   * @param checkpointManager The checkpoint manager
   */
  Model attachCheckpointManager(CheckpointManager checkpointManager);

  /**
   * Tells the layers who their parents are, allocates memory for
   * the weights, biases and activations and populates the with
   * initial data according to their respective initializers.
   * *
   * Call Chain should look like this:
   * - setParentLayer()
   * - Dense/Input/RNN.initialize()
   * - Layer.setWeightsPerUnit()
   * - Layer.initialize()
   * - initializeValues()
   */
  Model initialize();

  void forwardPass(float[] input);

  /**
   * Returns the error between the activations of the last layer and the label.
   *
   * @param label Target output.
   * @return Error vector (output - label).
   */
  float[] computeError(float[] label);

  /**
   * Applies the loss function to the label and the activations of the last layer.
   *
   * @param label Target output.
   * @return Loss value (scalar),
   */
  float computeLoss(float[] label);

  /**
   * @return Activations of the last layer.
   */
  float[] getOutput();

  Model train(TrainingDataPreprocessor preprocessor, int nEpochs, float learningRate);

  /**
   * Writes metrics collected during training to disk.
   */
  Model commitMetrics();

  /**
   * Serializes the current state of the network and writes it to disk.
   */
  Model commitCheckpoint();

  /**
   * Flattens a float[][] array.
   * @param input Nested float array of the form float[][]
   * @return Flattened array of the form float[]
   */
  static float[] floatFlat(float[][] input) {
    final int cols = input[0].length;
    float[] flatArray = new float[input.length * cols];

    for (int i = 0; i < input.length; i++) {
      System.arraycopy(input[i], 0, flatArray, i * cols, cols);
    }

    return flatArray;
  }

  String getName();
}
