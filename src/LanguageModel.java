
/**
 * An extension of the Model class which handles some of the language-model-specific configuration and setup.
 * Also provides a method generateOutput() which simplifies prompting the model.
 */
public class LanguageModel extends Model {

//  public String generateOutput(String prompt, int outputLength) {
//    if (prompt.length() != 40) return "Prompt length not 40 chars";
//
//    StringBuilder sb = new StringBuilder();
//
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
//  }
}
