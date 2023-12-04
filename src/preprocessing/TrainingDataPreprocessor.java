package preprocessing;

import preprocessing.batching.Batcher;
import preprocessing.datasources.TextSource;
import preprocessing.tokenization.TokenizationStrategy;
import preprocessing.vectorization.Sample;
import preprocessing.vectorization.VectorizationStrategy;

import java.io.Serializable;
import java.util.Iterator;

/**
 * Reads data from a text file, splits it into tokens, encodes those to vectors
 * and then collects the vectors into portions that can be passed into a network.
 */
public class TrainingDataPreprocessor extends Preprocessor implements Iterable<Sample[]>, Serializable {

  private final Batcher batcher;
  private final int batchSize;

  public TrainingDataPreprocessor(
    TextSource textSource,
    TokenizationStrategy tokenizationStrategy,
    VectorizationStrategy vectorizationStrategy,
    int stepOver,
    int batchSize
  ) {
    super(textSource, tokenizationStrategy, vectorizationStrategy, stepOver);

//    SampleAggregator sampleAggregator = new SampleAggregator(vectorizer, inputSize, stepOver);
    batcher = new Batcher(vectorizer, batchSize);
    this.batchSize = batchSize;
  }

  public int getBatchSize() {
    return batchSize;
  }

  public long getBatchCount() {
    return tokenReference.getTokenCount() / batchSize;
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
