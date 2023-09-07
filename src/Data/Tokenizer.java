package Data;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Reads words from a data source (PlainTextReader), builds a reference of
 * tokens and words, and splits the input text into a sequence of token numbers.
 */
public class Tokenizer {
    private final String delimiter;
    private final PlainTextReader reader;
    private int[] tokenizedText;
    private int tokenReferenceSize;
    HashMap<Integer, String> tokenReference;

    public Tokenizer(PlainTextReader reader, String delimiter) {
        this.reader = reader;
        this.delimiter = delimiter;
    }

    /**
     * Builds a dictionary which relates tokens (numbers) with words (strings).
     * And converts the input data into an array of tokens.
     */
    public void run() {
        HashMap<String, Integer> tokenOccurrences = new HashMap<>();
        ArrayList<String> words = new ArrayList<>();

        while (true) {
            // Read a line from the dataset
            String line = reader.readLine();

            // EOF reached
            if (line == null) break;

            // Split line into words and punctuation characters.
            Arrays
                    .stream(line.toLowerCase().split(delimiter))
//                    .map(String::trim)
                    .filter(word -> word.length() > 0)
                    .forEach(word -> {
                        words.add(word);
                        tokenOccurrences.putIfAbsent(word, 0);
                        tokenOccurrences.put(word, tokenOccurrences.get(word) + 1);
                    });
        }

        tokenReference = new HashMap<>();
        tokenizedText = new int[words.size()];

        HashMap<String, Integer> tokenBackReference = new HashMap<>();

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

        for (int i = 0; i < tokenizedText.length; i++) {
            tokenizedText[i] = tokenBackReference.get(words.get(i));
        }
    }

    /**
     * Returns the tokenized text.
     * @return an integer array where each integer represents a word from a dictionary.
     */
    public int[] getTokenizedText() {
        return tokenizedText;
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
}
