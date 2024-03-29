package preprocessing.tokenization;

import preprocessing.datasources.TextSource;

import java.util.Arrays;
import java.util.Iterator;

public class CachedTokenizer extends Tokenizer {
  public CachedTokenizer(TextSource textSource, TokenizationStrategy strategy) {
    super(textSource, strategy);
  }

  @Override
  public Iterator<Integer> iterator() {
    // TODO: Fix this
    final int[] tokens = new int[(int)getTokenReference().getTokenCount()];
    Iterator<Integer> iterator = super.iterator();

    for (int i = 0; i < tokens.length; i++) {
      tokens[i] = iterator.next();
    }

    return Arrays.stream(tokens).iterator();
  }
}
