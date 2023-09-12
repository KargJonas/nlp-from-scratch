package Data;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Reads words from a data source (PlainTextReader), builds a reference of
 * tokens and words, and splits the input text into a sequence of token numbers.
 */
public class Tokenizer implements Iterable<int[]> {
    private final String delimiter;
    private final Iterable<String> reader;
    private int tokenReferenceSize;
    HashMap<Integer, String> tokenReference;
    HashMap<String, Integer> tokenBackReference;

    public Tokenizer(Iterable<String> reader, String delimiter) {
        this.reader = reader;
        this.delimiter = delimiter;
    }

    /**
     * Builds a dictionary which relates tokens (numbers) with words (strings).
     * And converts the input data into an array of tokens.
     */
    public void buildTokenReference() {
        HashMap<String, Integer> tokenOccurrences = new HashMap<>();
        ArrayList<String> words = new ArrayList<>();

        for (String line : reader) {

            // Split line into words and punctuation characters.
            Arrays
                    .stream(line.toLowerCase().split(delimiter))
                    .filter(word -> !word.isEmpty())
                    .forEach(word -> {
                        words.add(word);
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

    @Override
    public Iterator<int[]> iterator() {
//        for (int i = 0; i < tokenizedText.length; i++) {
//            tokenizedText[i] = tokenBackReference.get(words.get(i));
//        }

        return new Iterator<>() {
            Iterator<String> fileIterator = reader.iterator();
            String line = fileIterator.next();



            @Override
            public boolean hasNext() {
                return false;
            }

            @Override
            public int[] next() {
                return new int[0];
            }
        };
    }
}
