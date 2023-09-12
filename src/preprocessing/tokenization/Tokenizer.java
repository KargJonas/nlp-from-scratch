package preprocessing.tokenization;

import preprocessing.datasources.TextSource;

import java.util.Iterator;


/**
 * Reads words from a data source (PlainTextReader), builds a reference of
 * tokens and words, and splits the input text into a sequence of token numbers.
 */
public class Tokenizer implements Iterable<Integer> {

  private final TokenReference tokenReference;
  private final TextSource textSource;
  private final TokenizationStrategy strategy;

  public Tokenizer(TextSource textSource, TokenizationStrategy strategy, TokenReference tokenReference) {
    this.textSource = textSource;
    this.strategy = strategy;
    this.tokenReference = tokenReference;
  }

  public Tokenizer(TextSource textSource, TokenizationStrategy strategy) {
    this.textSource = textSource;
    this.strategy = strategy;
    tokenReference = new TokenReference(textSource, this);
  }

  public String[] splitSectionByDelimiter(String section) {
    return strategy.apply(section);
  }

  @Override
  public Iterator<Integer> iterator() {
    return new Iterator<>() {
      final Iterator<String> sectionsIterator = textSource.iterator();
      int currentToken = 0;
      String[] section;

      {
        tokenizeNextSection();
      }

      private void tokenizeNextSection() {
        if (sectionsIterator.hasNext()) {
          currentToken = 0;
          section = splitSectionByDelimiter(sectionsIterator.next());
        }
      }

      @Override
      public boolean hasNext() {
        return currentToken < section.length || sectionsIterator.hasNext();
      }

      @Override
      public Integer next() {
        int token = tokenReference.encode(section[currentToken++]);
        if (currentToken >= section.length) tokenizeNextSection();
        return token;
      }
    };
  }

  public TokenReference getTokenReference() {
    return tokenReference;
  }
}
