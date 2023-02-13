package Data;

import java.util.*;
import java.util.stream.Collectors;

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

    public int[] getTokenizedText() {
        return tokenizedText;
    }

    public HashMap<Integer, String> getTokenReference() {
        return tokenReference;
    }

    public int getTokenReferenceSize() {
        return tokenReferenceSize;
    }
}
