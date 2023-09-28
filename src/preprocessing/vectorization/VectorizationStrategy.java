package preprocessing.vectorization;

import preprocessing.tokenization.TokenReference;

public interface VectorizationStrategy {

  float[] encode(int token);
  int decode(float[] vector);
  int getVectorSize();
  void setTokenReference(TokenReference tokenReference);

}
