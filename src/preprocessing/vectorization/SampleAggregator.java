package preprocessing.vectorization;

import java.util.Iterator;

/**
 * Collects and combines a number of vectors into a larger vector
 * that can then be passed into the network. This aggregation step
 * is necessary for networks that do not have memory.
 */
public class SampleAggregator implements Iterable<Sample> {

  Vectorizer vectorizer;
  int nInputTokens;
  int stepOver;

  public SampleAggregator(Vectorizer vectorizer, int nInputTokens, int stepOver) {
    this.vectorizer = vectorizer;
    this.nInputTokens = nInputTokens;
    this.stepOver = stepOver;
  }

  @Override
  public Iterator<Sample> iterator() {
    return new Iterator<>() {
      final Iterator<float[]> vectorIterator = vectorizer.iterator();
      boolean hasNext = true;

      final float[][] data = new float[nInputTokens][vectorizer.getVectorSize()];
      float[] label;

      {

        for (int i = 0; i < nInputTokens; i++) {
          if (!updateHasNext()) break;
          data[i] = vectorIterator.next();
        }

        if (updateHasNext()) {
          label = vectorIterator.next();
        }
      }

      private boolean updateHasNext() {
        if (!vectorIterator.hasNext()) {
          hasNext = false;
        }

        return hasNext;
      }

      // TODO: I hate this
      private void performAggregation() {
        System.arraycopy(data, stepOver, data, 0, nInputTokens - stepOver);

        data[data.length - stepOver] = label;

        for (int i = stepOver; i > 1; i--) {
          if (!updateHasNext()) break;
          data[data.length - i] = vectorIterator.next();
        }

        if (updateHasNext()) label = vectorIterator.next();
      }

      private Sample getSample() {
        float[][] newData = new float[nInputTokens][vectorizer.getVectorSize()];

        // TODO: Warning: the token vectors are heap elements, modifying them in one place will cause changes elsewhere.
        System.arraycopy(data, 0, newData, 0, nInputTokens);

        return new Sample(newData, label);
      }

      @Override
      public boolean hasNext() {
        return hasNext;
      }

      @Override
      public Sample next() {
        Sample sample = getSample();
        performAggregation();
        return sample;
      }
    };
  }
}
