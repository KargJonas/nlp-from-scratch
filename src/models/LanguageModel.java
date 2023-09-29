package models;

import models.Model;
import preprocessing.Preprocessor;
import preprocessing.PromptPreprocessor;

/**
 * An extension of the models.Model class which handles some of the language-model-specific configuration and setup.
 * Also provides a method generateOutput() which simplifies prompting the model.
 */
public class LanguageModel extends Model<LanguageModel> {

  Preprocessor preprocessor;

  public LanguageModel(String name) {
    super(name);
  }

  public LanguageModel(Model model) {
    super(model.name);
    this.trainingMonitor = model.trainingMonitor;
    this.checkpointManager = model.checkpointManager;
    this.layers = model.layers;
    this.lossFunction = model.lossFunction;
    this.checkpointNumber = model.checkpointNumber;
  }

  public LanguageModel setPreprocessor(Preprocessor preprocessor) {
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
      float[] flatEncodedPrompt = floatFlat(encodedPrompt);

      forwardPass(flatEncodedPrompt);
      float[] prediction = getOutput();
      stringBuilder.append(preprocessor.decode(prediction));
      System.arraycopy(encodedPrompt, 1, encodedPrompt, 0, encodedPrompt.length - 1);
      encodedPrompt[encodedPrompt.length - 1] = prediction.clone();
    }

    return stringBuilder.toString();
  }
}
