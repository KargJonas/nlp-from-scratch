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
//        Tokenizer tokenizer = new Tokenizer(reader, "\\W");
        Tokenizer tokenizer = new Tokenizer(reader, "(?!^)");
        Preprocessor preprocessor = new Preprocessor(tokenizer,40, 3);
        Preprocessor labelPreprocessor = new Preprocessor(tokenizer, 1, 3);
        tokenizer.run();

        // Should I make all of this "chainable" i.o.W.: should the Preprocessor also only read the tokens from
        // The Tokenizer bit-by-bit? This way the net could also read bit-by-bit from the Preprocessor while training.
        // This would mean unlimited dataset sizes.

        Preprocessor.FlatOneHotSentenceProvider oneHotSentenceProvider = preprocessor
                .getFlatOneHotSentenceProvider(0);

        Preprocessor.FlatOneHotSentenceProvider oneHotLabelProvider = labelPreprocessor
                .getFlatOneHotSentenceProvider(40);

        InputLayer il = InputLayer.build(Shape.build(40, tokenizer.getTokenReferenceSize()));
        DenseLayer dl0 = DenseLayer.build(128);
        DenseLayer dl1 = DenseLayer.build(tokenizer.getTokenReferenceSize())
                .setActivationFn(Activations.SOFTPLUS);
        SoftmaxLayer sml = SoftmaxLayer.build(tokenizer.getTokenReferenceSize());

        Model m = new Model()
                .addLayer(il)
                .addLayer(dl0)
                .addLayer(dl1)
                .addLayer(sml)
                .setErrorFunction(ErrorFunctions.CATEGORICAL_CROSSENTROPY)
                .initialize();

        double[] flattenedSentence = oneHotSentenceProvider.get();
        double[] label = oneHotLabelProvider.get();
        m.forwardPass(flattenedSentence);

        System.out.print("Prediction: ");
        System.out.println(Arrays.toString(m.getOutput()));

        System.out.print("Label: ");
        System.out.println(Arrays.toString(label));

        System.out.print("Cost: ");
        System.out.println(m.computeCost(label));
    }
}
