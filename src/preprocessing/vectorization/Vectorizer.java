package preprocessing.vectorization;

import preprocessing.tokenization.Tokenizer;

import java.util.Iterator;

/**
 * Converts tokens (integers which represent tokens) to vectors (float arrays).
 */
public class Vectorizer implements Iterable<float[]> {

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
  public int decode(float[] vector) {
    return strategy.decode(vector);
  }

  @Override
  public Iterator<float[]> iterator() {
    return new Iterator<>() {
      final Iterator<Integer> tokenIterator = tokenizer.iterator();

      @Override
      public boolean hasNext() {
        return tokenIterator.hasNext();
      }

      @Override
      public float[] next() {
        return strategy.encode(tokenIterator.next());
      }
    };
  }
}
