package preprocessing.vectorization;

import preprocessing.tokenization.TokenReference;

import java.io.Serializable;

public interface VectorizationStrategy extends Serializable {

  float[] encode(int token);
  int decode(float[] vector);
  int getVectorSize();
  void setTokenReference(TokenReference tokenReference);

}
