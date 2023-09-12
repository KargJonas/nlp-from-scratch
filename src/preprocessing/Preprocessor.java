package preprocessing;

import preprocessing.batching.Batcher;
import util.Shape;
import preprocessing.datasources.TextSource;
import preprocessing.tokenization.TokenReference;
import preprocessing.tokenization.TokenizationStrategy;
import preprocessing.tokenization.Tokenizer;
import preprocessing.vectorization.Sample;
import preprocessing.vectorization.SampleAggregator;
import preprocessing.vectorization.VectorizationStrategy;
import preprocessing.vectorization.Vectorizer;

import java.util.Iterator;

/**
 * Reads data from a text file, splits it into tokens, encodes those to vectors
 * and then collects the vectors into portions that can be passed into a network.
 */
public class Preprocessor implements Iterable<Sample[]> {

  public Tokenizer tokenizer;
  public TokenReference tokenReference;
  public Vectorizer vectorizer;
  SampleAggregator sampleAggregator;
  Batcher batcher;
  Shape inputShape;
  int batchSize;

  public Preprocessor(
    TextSource textSource,
    TokenizationStrategy tokenizationStrategy,
    VectorizationStrategy vectorizationStrategy,
    int inputSize,
    int stepOver,
    int batchSize
  ) {
    tokenizer = new Tokenizer(textSource, tokenizationStrategy);
    tokenReference = tokenizer.getTokenReference();
    vectorizationStrategy.setTokenReference(tokenReference);
    vectorizer = new Vectorizer(tokenizer, vectorizationStrategy);
    sampleAggregator = new SampleAggregator(vectorizer, inputSize, stepOver);
    batcher = new Batcher(sampleAggregator, batchSize);
    inputShape = Shape.build(inputSize, tokenReference.getTokenReferenceSize());
    this.batchSize = batchSize;
  }

  public Shape getInputShape() {
    return inputShape;
  }

  public int getTokenReferenceSize() {
    return tokenReference.getTokenReferenceSize();
  }

  public double getBatchSize() {
    return batchSize;
  }

  public String decode(double[] vector) {
    return tokenReference.decode(vectorizer.decode(vector));
  }

  @Override
  public Iterator<Sample[]> iterator() {
    return new Iterator<>() {
      final Iterator<Sample[]> batchIterator = batcher.iterator();

      @Override
      public boolean hasNext() {
        return batchIterator.hasNext();
      }

      @Override
      public Sample[] next() {
        return batchIterator.next();
      }
    };
  }
}
