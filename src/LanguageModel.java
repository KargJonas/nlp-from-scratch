import preprocessing.Preprocessor;
import preprocessing.datasources.StringReader;
import preprocessing.BasicPreprocessor;
import preprocessing.vectorization.Sample;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * An extension of the Model class which handles some of the language-model-specific configuration and setup.
 * Also provides a method generateOutput() which simplifies prompting the model.
 */
public class LanguageModel extends Model {

  public String generateOutput(String prompt, int outputLength, Preprocessor preprocessor) {
    if (prompt.length() != 40) return "Prompt length not 40 chars";

    StringReader  stringReader = new StringReader(prompt, 40);
    StringBuilder sb = new StringBuilder();
    Deque<float[]> deque = new ArrayDeque<>();

//    for (String section : stringReader) {
//      System.out.println("##" + section);
//      deque.add()
//    }

    BasicPreprocessor inputPreProc = new BasicPreprocessor(stringReader, preprocessor);

    for (Sample sample : inputPreProc) {
      System.out.println(preprocessor.decode(sample));
    }

//    System.out.println(inputPreProc);

//    float[][] vectorizedInput = preprocessor.
//
//    for (int i = 0; i < outputLength; i++) {
//      double[] input = oneHotSentenceProvider.get();
//      forwardPass(input);
//      double[] outputVector = m.getOutput();
//
//      int tokenIndex = Preprocessor.getTokenFromOneHot(outputVector);
//      String token = tokenizer.decode(tokenIndex);
//
//      sb.append(token);
//    }
//
//    return sb.toString();

    return "";
  }
}
