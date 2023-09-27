package preprocessing.tokenization;

import preprocessing.datasources.TextSource;

import java.util.*;
import java.util.stream.Collectors;

public class TokenReference {
  HashMap<Integer, String> tokenReference;
  HashMap<String, Integer> tokenBackReference;
  private int tokenReferenceSize = 0;

  private final TextSource textSource;
  private final Tokenizer tokenizer;

  long wordCount = 0;

  public TokenReference(TextSource textSource, Tokenizer tokenizer) {
    this.textSource = textSource;
    this.tokenizer = tokenizer;

    buildTokenReference();
  }

  /**
   * Builds a dictionary which relates tokens (numbers) with words (strings).
   */
  public void buildTokenReference() {
    HashMap<String, Integer> tokenOccurrences = new HashMap<>();

    for (String section : textSource) {

      // Split section into words and punctuation characters.
      Arrays
        .stream(tokenizer.splitSectionByDelimiter(section))
        .filter(word -> !word.isEmpty())
        .forEach(word -> {
          wordCount++;
          tokenOccurrences.putIfAbsent(word, 0);
          tokenOccurrences.put(word, tokenOccurrences.get(word) + 1);
        });
    }

    // Maps for translating back and forth
    tokenReference = new HashMap<>();
    tokenBackReference = new HashMap<>();

    // This list contains an ordered (by occurrence, descending) version of all tokens.
    ArrayList<String> orderedTokenList = tokenOccurrences.entrySet()
      .stream()
      .sorted(Comparator.comparing(Map.Entry<String, Integer>::getValue).reversed())
      .map(Map.Entry::getKey)
      .collect(Collectors.toCollection(ArrayList::new));

    tokenReferenceSize = orderedTokenList.size();

    for (int i = 0; i < orderedTokenList.size(); i++) {
      String word = orderedTokenList.get(i);

      tokenReference.put(i, word);
      tokenBackReference.put(word, i);
    }
  }

  public long getTokenCount() {
    return wordCount;
  }

  /**
   * Returns the token dictionary.
   * @return a HashMap which relates a token (integer) to a word (string).
   */
  public HashMap<Integer, String> getTokenReference() {
    return tokenReference;
  }

  /**
   * Returns the number of distinct words in the token reference.
   * @return size of the token reference.
   */
  public int getTokenReferenceSize() {
    return tokenReferenceSize;
  }

  /**
   * Converts a token (number) to a word (string).
   * @param token Token reference number
   * @return Word
   */
  public String decode(int token) {
    return tokenReference.get(token);
  }

  /**
   * Converts a word (string) to a token (number).
   * @param word Word
   * @return Token reference number
   */
  public int encode(String word) {
    return tokenBackReference.get(word);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();

    for (Map.Entry<Integer, String> entry : tokenReference.entrySet()) {
      sb.append(String.format("%s: %s\n", entry.getKey(), entry.getValue().replace("\n", "\\n")));
    }

    return sb.toString();
  }
}
