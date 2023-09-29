package preprocessing.batching;

import preprocessing.vectorization.Sample;
import preprocessing.vectorization.SampleAggregator;

import java.io.Serializable;
import java.util.Iterator;

public class Batcher implements Iterable<Sample[]>, Serializable {

  SampleAggregator sampleAggregator;
  int batchSize;

  public Batcher(SampleAggregator sampleAggregator, int batchSize) {
    this.sampleAggregator = sampleAggregator;
    this.batchSize = batchSize;
  }

  @Override
  public Iterator<Sample[]> iterator() {
    return new Iterator<>() {
      final Iterator<Sample> sampleIterator = sampleAggregator.iterator();
      Sample[] batch;
      boolean hasNext = true;

      {
        buildBatch();
      }

      private void buildBatch() {
        batch = new Sample[batchSize];

        for (int i = 0; i < batchSize; i++) {
          if (!updateHasNext()) break;
          batch[i] = sampleIterator.next();
        }
      }

      private boolean updateHasNext() {
        if (!sampleIterator.hasNext()) {
          hasNext = false;
        }

        return hasNext;
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
