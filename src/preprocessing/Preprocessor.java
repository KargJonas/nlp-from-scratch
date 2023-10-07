package preprocessing;

import util.Shape;
import preprocessing.datasources.TextSource;
import preprocessing.tokenization.TokenReference;
import preprocessing.tokenization.TokenizationStrategy;
import preprocessing.tokenization.Tokenizer;
import preprocessing.vectorization.Sample;
import preprocessing.vectorization.VectorizationStrategy;
import preprocessing.vectorization.Vectorizer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * Reads data from a text file, splits it into tokens, encodes those to vectors
 * and then collects the vectors into portions that can be passed into a network.
 */
public class Preprocessor implements Serializable {

  protected TokenizationStrategy tokenizationStrategy;
  protected VectorizationStrategy vectorizationStrategy;

  public Tokenizer tokenizer;
  public TokenReference tokenReference;
  public Vectorizer vectorizer;

  int inputSize;
  int stepOver;

  public Preprocessor(
    TextSource textSource,
    TokenizationStrategy tokenizationStrategy,
    VectorizationStrategy vectorizationStrategy,
    int stepOver
  ) {
    this.tokenizationStrategy = tokenizationStrategy;
    this.vectorizationStrategy = vectorizationStrategy;

    tokenizer = new Tokenizer(textSource, tokenizationStrategy);
    tokenReference = tokenizer.getTokenReference();
    vectorizationStrategy.setTokenReference(tokenReference);
    vectorizer = new Vectorizer(tokenizer, vectorizationStrategy);
    inputSize = tokenReference.getTokenReferenceSize();

    this.stepOver = stepOver;
  }

  public Preprocessor(
    TextSource textSource,
    Preprocessor preprocessor
  ) {
    // Use config directly from provided preprocessor.
    tokenizationStrategy  = preprocessor.tokenizationStrategy;
    vectorizationStrategy = preprocessor.vectorizationStrategy;
    stepOver  = preprocessor.stepOver;

    // TODO: Don't forget that tokenReference.wordCount still refers to training data word count
    tokenReference = preprocessor.tokenReference;

    // Create a new tokenizer with the provided text source but old tokenReference
    tokenizer        = new Tokenizer(textSource, preprocessor.tokenizationStrategy, tokenReference);
    vectorizer       = new Vectorizer(tokenizer, vectorizationStrategy);

    inputSize = tokenReference.getTokenReferenceSize();
  }

  public int getInputSize() {
    return inputSize;
  }

  public int getTokenReferenceSize() {
    return tokenReference.getTokenReferenceSize();
  }

  public long getTokenCount() {
    return tokenReference.getTokenCount();
  }

  public String decode(float[] vector) {
    return tokenReference.decode(vectorizer.decode(vector)).replace("\n", "\\n");
  }

  public String decode(float[][] vector) {
    return Arrays.stream(vector).map(this::decode).collect(Collectors.joining());
  }

  public String decode(Sample sample) {
    return String.format("%s -> %s", decode(sample.data()), decode(sample.label()));
  }
}
