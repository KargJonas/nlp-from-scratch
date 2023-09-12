package preprocessing.tokenization;

/**
 * Splits a string into an array of strings where each string contains exactly one character.
 */
public class CharTokenizer implements TokenizationStrategy {

  public String[] apply(String section) {
    String[] tokens = new String[section.length()];

    for (int i = 0; i < section.length(); i++)
      tokens[i] = String.valueOf(section.charAt(i));

    return tokens;
  }

  public static TokenizationStrategy build() {
    return new CharTokenizer();
  }
}
