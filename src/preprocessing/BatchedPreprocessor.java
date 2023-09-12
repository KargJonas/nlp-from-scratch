package preprocessing;

import preprocessing.batching.Batcher;
import preprocessing.datasources.TextSource;
import preprocessing.tokenization.TokenizationStrategy;
import preprocessing.vectorization.Sample;
import preprocessing.vectorization.VectorizationStrategy;

import java.util.Iterator;

/**
 * Reads data from a text file, splits it into tokens, encodes those to vectors
 * and then collects the vectors into portions that can be passed into a network.
 */
public class BatchedPreprocessor extends Preprocessor implements Iterable<Sample[]> {

  Batcher batcher;
  int batchSize;

  public BatchedPreprocessor(
    TextSource textSource,
    TokenizationStrategy tokenizationStrategy,
    VectorizationStrategy vectorizationStrategy,
    int inputSize,
    int stepOver,
    int batchSize
  ) {
    super(textSource, tokenizationStrategy, vectorizationStrategy, inputSize, stepOver);

    batcher = new Batcher(sampleAggregator, batchSize);
    this.batchSize = batchSize;
  }

  public double getBatchSize() {
    return batchSize;
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
