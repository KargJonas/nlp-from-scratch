package preprocessing;

import preprocessing.datasources.StringReader;
import preprocessing.datasources.TextSource;

import java.util.Iterator;

public class PromptPreprocessor {

  Preprocessor parentPreprocessor;

  public PromptPreprocessor(Preprocessor preprocessor) {
    this.parentPreprocessor = preprocessor;
  }

  public float[][] encode(String prompt) {
    TextSource textSource = new StringReader(prompt, prompt.length());
    Preprocessor preprocessor = new Preprocessor(textSource, parentPreprocessor);
    Iterator<float[]> vectorIterator = preprocessor.vectorizer.iterator();
    float[][] vector = new float[preprocessor.inputSize][preprocessor.getTokenReferenceSize()];

    for (int i = 0; i < preprocessor.inputSize; i++) {
      vector[i] = vectorIterator.next();
    }

    return vector;
  }
}
