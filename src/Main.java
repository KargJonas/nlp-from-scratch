import Data.PlainTextReader;
import Data.Preprocessor;
import Data.Tokenizer;
import Layers.DenseLayer;
import Layers.InputLayer;
import Layers.SoftmaxLayer;
import Util.Activations;
import Util.ErrorFunctions;
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

//        Preprocessor.OneHotSentenceProvider oneHotSentenceProvider = preprocessor
//                .getOneHotSentenceProvider();

        Preprocessor.FlatOneHotSentenceProvider oneHotSentenceProvider = preprocessor
                .getFlatOneHotSentenceProvider();

        InputLayer il = InputLayer.build(Shape.build(16, 16));
        DenseLayer dl0 = DenseLayer.build(128);
        DenseLayer dl1 = DenseLayer.build(10).setActivationFn(Activations.SOFTPLUS);
        SoftmaxLayer sml = SoftmaxLayer.build(10);

        Model m = new Model()
                .addLayer(il)
                .addLayer(dl0)
                .addLayer(dl1)
                .addLayer(sml)
                .setErrorFunction(ErrorFunctions.CATEGORICAL_CROSSENTROPY)
                .initialize();



        double[] flattenedSentence = oneHotSentenceProvider.get();
        double[] output = m.forwardPass(flattenedSentence);

        System.out.println(Arrays.toString(output));
    }
}
