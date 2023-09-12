package preprocessing.vectorization;

import preprocessing.tokenization.Tokenizer;

import java.util.Iterator;

/**
 * Converts tokens (integers which represent tokens) to vectors (float arrays).
 */
public class Vectorizer implements Iterable<double[]> {

  VectorizationStrategy strategy;
  Tokenizer tokenizer;

  public Vectorizer(Tokenizer tokenizer, VectorizationStrategy strategy) {
    this.tokenizer = tokenizer;
    this.strategy = strategy;
  }

  public int getVectorSize() {
    return strategy.getVectorSize();
  }

  // TODO: Find a real solution for backwards vectorization
  public int decode(double[] vector) {
    return strategy.decode(vector);
  }

  @Override
  public Iterator<double[]> iterator() {
    return new Iterator<>() {
      final Iterator<Integer> tokenIterator = tokenizer.iterator();

      @Override
      public boolean hasNext() {
        return tokenIterator.hasNext();
      }

      @Override
      public double[] next() {
        return strategy.encode(tokenIterator.next());
      }
    };
  }
}
