import layers.DenseLayer;
import layers.InputLayer;
import layers.SoftmaxLayer;
import util.Activations;
import util.LossFunctions;
import util.Shape;
import preprocessing.Preprocessor;
import preprocessing.datasources.TextFileReader;
import preprocessing.tokenization.CharTokenizer;
import preprocessing.vectorization.OneHotVectorizer;
import preprocessing.vectorization.Sample;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        var reader       = new TextFileReader("/home/jonas/code/nlp-from-scratch/src/bla.txt", 10);
        var preprocessor = new Preprocessor(reader, CharTokenizer.build(), OneHotVectorizer.build(), 40, 1, 5);
        var model        = new LanguageModel();

        var nTokens = preprocessor.getNTokens();

        for (Sample[] sample : preprocessor) {
            System.out.println(Arrays.toString(sample));
        }

////        var sampleProvider = samplePreprocessor.getFlatOneHotSentenceProvider(0);
////        var labelProvider  = labelPreprocessor.getFlatOneHotSentenceProvider(40);
////
//
//        var il = InputLayer.build(Shape.build(40, nTokens));
//        var dl0 = DenseLayer.build(128);
//        var dl1 = DenseLayer.build(nTokens)
//                .setActivationFn(Activations.SOFTPLUS);
//        var sml = SoftmaxLayer.build(nTokens);
//
//        model
//                .addLayer(il)
//                .addLayer(dl0)
//                .addLayer(dl1)
//                .addLayer(sml)
//                .setLossFunction(LossFunctions.CATEGORICAL_CROSSENTROPY)
//                .initialize()
//                .train(preprocessor, 20, 10, 0.05);

    }
}
