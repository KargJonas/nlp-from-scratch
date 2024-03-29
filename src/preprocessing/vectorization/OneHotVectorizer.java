package preprocessing.vectorization;

import preprocessing.tokenization.TokenReference;

/**
 * Provides methods for converting tokens to one-hot vectors and vice versa.
 */
public class OneHotVectorizer implements VectorizationStrategy {

  TokenReference tokenReference;

  @Override
  public float[] encode(int token) {
    float[] vector = new float[tokenReference.getTokenReferenceSize()];
    vector[token] = 1;
    return vector;
  }

  @Override
  public int decode(float[] vector) {
    int likeliestToken = 0;

    for (int i = 0; i < vector.length; i++) {
      if (vector[i] > vector[likeliestToken]) likeliestToken = i;
    }

    return likeliestToken;
  }

  @Override
  public int getVectorSize() {
    return tokenReference.getTokenReferenceSize();
  }

  @Override
  public void setTokenReference(TokenReference tokenReference) {
    this.tokenReference = tokenReference;
  }

  public static OneHotVectorizer build() {
    return new OneHotVectorizer();
  }
}
