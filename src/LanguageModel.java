import preprocessing.Preprocessor;
import preprocessing.PromptPreprocessor;
import preprocessing.datasources.StringReader;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

/**
 * An extension of the Model class which handles some of the language-model-specific configuration and setup.
 * Also provides a method generateOutput() which simplifies prompting the model.
 */
public class LanguageModel extends Model {

  Preprocessor preprocessor;

  public LanguageModel(Preprocessor preprocessor) {
    this.preprocessor = preprocessor;
  }

  public String generateOutput(String prompt, int outputLength) {
    if (prompt.length() > 40) return "Max prompt size exceeded";

    // TODO: If the original dataset does not include space, this will cause errors.
    prompt = " ".repeat(40 - prompt.length()) + prompt;

    var stringBuilder = new StringBuilder();
    var promptPreprocessor = new PromptPreprocessor(preprocessor);
    double[][] encodedPrompt = promptPreprocessor.encode(prompt);

    for (int i = 0; i < outputLength; i++) {
      double[] flatEncodedPrompt = Arrays.stream(encodedPrompt)
        .flatMapToDouble(Arrays::stream)
        .toArray();

      forwardPass(flatEncodedPrompt);
      double[] prediction = getOutput();
      stringBuilder.append(preprocessor.decode(prediction));
      System.arraycopy(encodedPrompt, 1, encodedPrompt, 0, encodedPrompt.length - 1);
      encodedPrompt[encodedPrompt.length - 1] = prediction.clone();
    }

    return stringBuilder.toString();
  }
}
