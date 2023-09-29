package preprocessing.tokenization;

import java.io.Serializable;

/**
 * Splits a section of text into tokens.
 * NOTE: In most cases, this cannot be naively done using a .split() operation,
 *   because the provided sections have an arbitrary cutoff, so `section` may
 *   contain partial tokens.
 */
public interface TokenizationStrategy extends Serializable {

  String[] apply(String section);
}
