package preprocessing.batching;

import preprocessing.vectorization.Sample;
import preprocessing.vectorization.Vectorizer;

import java.io.Serializable;
import java.util.Iterator;

public class Batcher implements Iterable<Sample[]>, Serializable {

  Vectorizer vectorizer;
  int batchSize;

  public Batcher(Vectorizer vectorizer, int batchSize) {
    this.vectorizer = vectorizer;
    this.batchSize = batchSize;
  }

  @Override
  public Iterator<Sample[]> iterator() {
    return new Iterator<>() {
      final Iterator<float[]> vectorIterator = vectorizer.iterator();
      Sample[] batch;
      boolean hasNext = true;

      {
        buildBatch();
      }

      private void buildBatch() {
        batch = new Sample[batchSize];

        for (int i = 0; i < batchSize; i++) {
          float[] data = vectorIterator.next();
          if (updateHasNext()) break;

          float[] label = vectorIterator.next();
          if (updateHasNext()) break;

          batch[i] = new Sample(data, label);
        }
      }

      private boolean updateHasNext() {
        if (!vectorIterator.hasNext()) {
          hasNext = false;
        }

        return !hasNext;
      }

      @Override
      public boolean hasNext() {
        return hasNext;
      }

      @Override
      public Sample[] next() {
        Sample[] newBatch = batch;
        buildBatch();
        return newBatch;
      }
    };
  }
}
