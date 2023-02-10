import Data.PlainTextReader;
import Data.Tokenizer;
import Layers.DenseLayer;
import Layers.InputLayer;
import Util.Activation;
import Util.Shape;

public class Main {
    public static void main(String[] args) {
        PlainTextReader reader = new PlainTextReader("/home/jonas/code/nlp-1/src/nietzsche.txt");
        Tokenizer tokenizer = new Tokenizer(reader);

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
        DenseLayer dl1 = DenseLayer.build(10).setActivationFn(Activation.SIGMOID);

        Model m = new Model()
                .addLayer(il)
                .addLayer(dl0)
                .addLayer(dl1)
                .initialize();
    }
}