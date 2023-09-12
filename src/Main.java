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

public class Main {
    public static void main(String[] args) {
        int inputSize = 40;

        var reader       = new TextFileReader("/home/jonas/code/nlp-from-scratch/src/bla.txt", 10);
        var preprocessor = new Preprocessor(reader, CharTokenizer.build(), OneHotVectorizer.build(), inputSize, 1, 1);
        var model        = new LanguageModel();

        var tokenRefSize = preprocessor.getTokenReferenceSize();

        var il = InputLayer.build(Shape.build(inputSize, tokenRefSize));
        var dl0 = DenseLayer.build(128);
        var dl1 = DenseLayer.build(tokenRefSize)
                .setActivationFn(Activations.SOFTPLUS);
        var sml = SoftmaxLayer.build(tokenRefSize);

        model
                .addLayer(il)
                .addLayer(dl0)
                .addLayer(dl1)
                .addLayer(sml)
                .setLossFunction(LossFunctions.CATEGORICAL_CROSSENTROPY)
                .initialize()
                .train(preprocessor, 20000, 0.01);




//        for (Sample[] batch : preprocessor) {
//
//            for (Sample sample : batch) {
//
//                double[][] data = sample.data();
//                double[] label = sample.label();
//
//                StringBuilder sb = new StringBuilder();
//
//                for (double[] token : data) {
//                    sb.append(preprocessor.decode(token));
//                }
//
//                sb.append(" -> ");
//                sb.append(preprocessor.decode(label));
//
//                System.out.println(sb);
//            }
//        }
    }
}
