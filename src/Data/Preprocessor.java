package Data;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

public class Preprocessor {
    public interface SentenceProvider {
        int[] get();
    }

    public interface OneHotProvider {
        double[] get(int token);
    }

    public interface OneHotSentenceProvider {
        double[][] next();
    }

    public interface FlatOneHotSentenceProvider {
        double[] next();
    }

    private final int maxSentenceLength;
    private final int stepOver;
    private final Tokenizer tokenizer;

    /**
     * Instantiate a new Preprocessor
     * @param maxSentenceLength Max sentence length. (Last sentence might be shorter)
     * @param stepOver Determines how much overlap there is between two sentences.
     *                 If stepOver ist the same value as maxSentenceLength,
     *                 then there will be no overlap.
     */
    public Preprocessor(Tokenizer tokenizer, int maxSentenceLength, int stepOver) {
        this.maxSentenceLength = maxSentenceLength;
        this.stepOver = stepOver;
        this.tokenizer = tokenizer;
    }

    public SentenceProvider getSentenceProvider(int[] tokens, int initialIndex) {
        AtomicInteger currentToken = new AtomicInteger(initialIndex);

        /*
         * From token list get subarray starting at currentToken until (currentToken + maxSentenceLength) or
         * EOF, whichever is smaller. This prevents returning arrays with partially empty array elements.
         * This is also the reason why maxSentenceLength is called the way it is called.
         */
        return () -> {
            if (currentToken.get() - tokens.length == 0) return null;

            return Arrays.copyOfRange(
              tokens,
              Math.min(currentToken.get(), tokens.length),
              Math.min(currentToken.getAndAdd(stepOver) + maxSentenceLength, tokens.length)
            );
        };
    }

    public OneHotProvider getOneHotProvider(int tokenReferenceSize) {
        return (token) -> {
            double[] oneHotVector = new double[tokenReferenceSize];
            oneHotVector[token] = 1;
            return oneHotVector;
        };
    }

    public static int getTokenFromOneHot(double[] oneHotVector) {
        int highestIndex = 0;

        // Find vector component with the highest activation
        for (int i = 0; i < oneHotVector.length; i++) {
            if (oneHotVector[i] > oneHotVector[highestIndex]) highestIndex = i;
        }

        return highestIndex;
    }

    public OneHotSentenceProvider getOneHotSentenceProvider(int initialIndex) {
        Preprocessor.SentenceProvider sentenceProvider = getSentenceProvider(tokenizer.getTokenizedText(), initialIndex);
        Preprocessor.OneHotProvider oneHotProvider = getOneHotProvider(tokenizer.getTokenReferenceSize());

        return () -> {
            int[] sentence = sentenceProvider.get();
            if (sentence == null) return null;

            double[][] oneHotSentence = new double[sentence.length][];

            for (int i = 0; i < sentence.length; i++) {
                oneHotSentence[i] = oneHotProvider.get(sentence[i]);
            }

            return oneHotSentence;
        };
    }

    public FlatOneHotSentenceProvider getFlatOneHotSentenceProvider(int initialIndex) {
        Preprocessor.SentenceProvider sentenceProvider = getSentenceProvider(tokenizer.getTokenizedText(), initialIndex);
        Preprocessor.OneHotProvider oneHotProvider = getOneHotProvider(tokenizer.getTokenReferenceSize());

        return () -> {
            int[] sentence = sentenceProvider.get();
            if (sentence == null) return null;

            double[] oneHotSentence = new double[maxSentenceLength * tokenizer.getTokenReferenceSize()];

            for (int i = 0; i < sentence.length; i++) {
                double[] oneHotToken = oneHotProvider.get(sentence[i]);
                System.arraycopy(oneHotToken, 0, oneHotSentence, i * oneHotToken.length, oneHotToken.length);
            }

            return oneHotSentence;
        };
    }
}
