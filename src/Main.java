import Data.PlainTextReader;
import Data.Preprocessor;
import Data.Tokenizer;
import Layers.DenseLayer;
import Layers.InputLayer;
import Util.Activation;
import Util.Shape;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        PlainTextReader reader = new PlainTextReader("/home/jonas/code/nlp-1/src/nietzsche.txt");
        Tokenizer tokenizer = new Tokenizer(reader, "\\W");
        Preprocessor preprocessor = new Preprocessor(tokenizer,40, 3);
        tokenizer.run();

        // Should I make all of this "chainable" i.o.W.: should the Preprocessor also only read the tokens from
        // The Tokenizer bit-by-bit? This way the net could also read bit-by-bit from the Preprocessor while training.
        // This would mean unlimited dataset sizes.

        Preprocessor.OneHotSentenceProvider oneHotSentenceProvider = preprocessor
                .getOneHotSentenceProvider();

//        for (int i = 0; i < 10; i++) {
//            System.out.println(Arrays.deepToString(oneHotSentenceProvider.get()));
//        }

        /*
         * model = keras.Sequential(
         *     [
         *         keras.Input(shape=(maxlen, len(chars))),
         *         layers.LSTM(128),
         *         layers.Dense(len(chars), activation="softmax"),
         *     ]
         * )
         */

        // Current thoughts:
        // Input is a 2d array - list of words in a sentence, each word in one-hot encoding.
        // Problem: Dimensionality of one-hot vectors grows proportionately to the size of the corpus.
        // => Inefficient in terms of memory
        // one-hot size can be reduced by choosing characters as smallest units instead of words.
        // => downside: potentially slower learning speeds, as understanding of sub-word structures is necessary

        InputLayer il = InputLayer.build(Shape.build(16, 16));
        DenseLayer dl0 = DenseLayer.build(128);
        DenseLayer dl1 = DenseLayer.build(10).setActivationFn(Activation.SOFTPLUS);

        Model m = new Model()
                .addLayer(il)
                .addLayer(dl0)
                .addLayer(dl1)
                .initialize();

        double[] flattenedSentence = Arrays.stream(oneHotSentenceProvider.get()).flatMapToDouble(Arrays::stream).toArray();
        double[] output = m.forwardPass(flattenedSentence);

        System.out.println(Arrays.toString(output));
    }
}