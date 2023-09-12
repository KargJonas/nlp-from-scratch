package preprocessing;

import preprocessing.Preprocessor;
import preprocessing.datasources.TextSource;
import preprocessing.tokenization.TokenizationStrategy;
import preprocessing.vectorization.Sample;
import preprocessing.vectorization.VectorizationStrategy;

import java.util.Iterator;

public class BasicPreprocessor extends Preprocessor implements Iterable<Sample> {

  public BasicPreprocessor(TextSource textSource, TokenizationStrategy tokenizationStrategy, VectorizationStrategy vectorizationStrategy, int inputSize, int stepOver) {
    super(textSource, tokenizationStrategy, vectorizationStrategy, inputSize, stepOver);
  }

  public BasicPreprocessor(TextSource textSource, Preprocessor preprocessor) {
    super(textSource, preprocessor);
  }

  @Override
  public Iterator<Sample> iterator() {
    return new Iterator<>() {
      final Iterator<Sample> sampleIterator = sampleAggregator.iterator();

      @Override
      public boolean hasNext() {
        return sampleIterator.hasNext();
      }

      @Override
      public Sample next() {
        return sampleIterator.next();
      }
    };
  }
}
