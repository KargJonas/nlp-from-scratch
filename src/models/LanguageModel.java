package models;

import checkpoint.CheckpointManager;
import layers.GenericLayer;
import preprocessing.PromptPreprocessor;
import preprocessing.TrainingDataPreprocessor;
import telemetry.TrainingMonitor;
import util.LossFunctions;

import java.io.Serializable;

/**
 * An extension of the models.Model class which handles some of the language-model-specific configuration and setup.
 * Also provides a method generateOutput() which simplifies prompting the model.
 */
public class LanguageModel implements Model, Serializable {

  TrainingDataPreprocessor preprocessor;
  Model model;

  public LanguageModel() {
    this.model = new GenericModel();
  }

  public LanguageModel(Model model) {
    this.model = model;
  }

  public LanguageModel setPreprocessor(TrainingDataPreprocessor preprocessor) {
    this.preprocessor = preprocessor;
    return this;
  }

  public String generateOutput(String prompt, int outputLength) {
    if (prompt.length() > 40) return "Max prompt size exceeded";

    // TODO: If the original dataset does not include space, this will cause errors.
    prompt = " ".repeat(40 - prompt.length()) + prompt;

    var stringBuilder = new StringBuilder();
    var promptPreprocessor = new PromptPreprocessor(preprocessor);
    float[][] encodedPrompt = promptPreprocessor.encode(prompt);

    for (int i = 0; i < outputLength; i++) {
      float[] flatEncodedPrompt = Model.floatFlat(encodedPrompt);

      forwardPass(flatEncodedPrompt);
      float[] prediction = getOutput();
      stringBuilder.append(preprocessor.decode(prediction));
      System.arraycopy(encodedPrompt, 1, encodedPrompt, 0, encodedPrompt.length - 1);
      encodedPrompt[encodedPrompt.length - 1] = prediction.clone();
    }

    return stringBuilder.toString();
  }

  @Override
  public CheckpointManager getCheckpointManager() {
    return model.getCheckpointManager();
  }

  @Override
  public LanguageModel addLayer(GenericLayer<?> layer) {
    model.addLayer(layer);
    return this;
  }

  @Override
  public LanguageModel setName(String name) {
    model.setName(name);
    return this;
  }

  @Override
  public LanguageModel setLossFunction(LossFunctions.LossFn lossFunction) {
    model.setLossFunction(lossFunction);
    return this;
  }

  @Override
  public LanguageModel attachTelemetry(TrainingMonitor trainingMonitor) {
    model.attachTelemetry(trainingMonitor);
    return this;
  }

  @Override
  public LanguageModel attachCheckpointManager(CheckpointManager checkpointManager) {
    model.attachCheckpointManager(checkpointManager);
    return this;
  }

  @Override
  public LanguageModel initialize() {
    model.initialize();
    return this;
  }

  @Override
  public void forwardPass(float[] input) {
    model.forwardPass(input);
  }

  @Override
  public float[] computeError(float[] label) {
    return model.computeError(label);
  }

  @Override
  public float computeLoss(float[] label) {
    return model.computeLoss(label);
  }

  @Override
  public float[] getOutput() {
    return model.getOutput();
  }

  public LanguageModel train(int nEpochs, float learningRate) {
    return this.train(preprocessor, nEpochs, learningRate);
  }

  @Override
  public LanguageModel train(TrainingDataPreprocessor preprocessor, int nEpochs, float learningRate) {
    model.train(preprocessor, nEpochs, learningRate);
    return this;
  }

  @Override
  public LanguageModel commitMetrics() {
    model.commitMetrics();
    return this;
  }

  @Override
  public LanguageModel createCheckpoint() {
    if (model.getCheckpointManager() == null) throw new RuntimeException("Cant create checkpoint: No checkpoint manager configured.");
    model.getCheckpointManager().createCheckpoint(this);
    return this;
  }

  @Override
  public String getName() {
    return model.getName();
  }
}
