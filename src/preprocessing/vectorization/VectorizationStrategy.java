package preprocessing.vectorization;

import preprocessing.tokenization.TokenReference;

public interface VectorizationStrategy {

  double[] encode(int token);
  int decode(double[] vector);
  int getVectorSize();
  void setTokenReference(TokenReference tokenReference);

}
