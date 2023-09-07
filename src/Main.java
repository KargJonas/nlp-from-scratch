import Data.PlainTextReader;
import Data.Preprocessor;
import Data.Tokenizer;
import Layers.DenseLayer;
import Layers.InputLayer;
import Layers.SoftmaxLayer;
import Util.Activations;
import Util.LossFunctions;
import Util.Shape;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
//        PlainTextReader reader = new PlainTextReader("/home/jonas/code/nlp-from-scratch/src/nietzsche.txt");
        PlainTextReader reader = new PlainTextReader("/home/jonas/code/nlp-from-scratch/src/bla.txt");
//        PlainTextReader reader = new PlainTextReader("/home/jonas/code/nlp-from-scratch/src/boot.txt");
//        PlainTextReader reader = new PlainTextReader("/home/jonas/code/nlp-from-scratch/src/mein_name.txt");
//        Tokenizer tokenizer = new Tokenizer(reader, "\\W");
        Tokenizer tokenizer = new Tokenizer(reader, "(?!^)");
        Preprocessor samplePreprocessor = new Preprocessor(tokenizer,40, 3);
        Preprocessor labelPreprocessor  = new Preprocessor(tokenizer, 1, 3);

        tokenizer.run();

        LanguageModel model = new LanguageModel();

        // Should I make all of this "chainable" i.o.W.: should the Preprocessor also only read the tokens from
        // The Tokenizer bit-by-bit? This way the net could also read bit-by-bit from the Preprocessor while training.
        // This would mean unlimited dataset sizes.

        var sampleProvider = samplePreprocessor.getFlatOneHotSentenceProvider(0);
        var labelProvider  = labelPreprocessor.getFlatOneHotSentenceProvider(40);

        InputLayer il = InputLayer.build(Shape.build(40, tokenizer.getTokenReferenceSize()));
        DenseLayer dl0 = DenseLayer.build(128);
        DenseLayer dl1 = DenseLayer.build(tokenizer.getTokenReferenceSize())
                .setActivationFn(Activations.SOFTPLUS);
        SoftmaxLayer sml = SoftmaxLayer.build(tokenizer.getTokenReferenceSize());

        model
                .addLayer(il)
                .addLayer(dl0)
                .addLayer(dl1)
                .addLayer(sml)
                .setLossFunction(LossFunctions.CATEGORICAL_CROSSENTROPY)
                .initialize()
                .train(sampleProvider, labelProvider, 20, 10, 0.05);


    }
}
