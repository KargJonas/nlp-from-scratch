import layers.DenseLayer;
import layers.InputLayer;
import layers.SoftmaxLayer;
import preprocessing.TrainingDataPreprocessor;
import util.Activations;
import preprocessing.datasources.TextFileReader;
import preprocessing.tokenization.CharTokenizer;
import preprocessing.vectorization.OneHotVectorizer;
import util.LossFunctions;

public class Main {
    public static void main(String[] args) {
        int inputSize = 40;

        var reader       = new TextFileReader("/home/jonas/code/nlp-from-scratch/assets/nietzsche.txt", 10);
        var preprocessor = new TrainingDataPreprocessor(reader, CharTokenizer.build(), OneHotVectorizer.build(), inputSize, 1, 20);
        var model        = new LanguageModel(preprocessor);

        var tokenRefSize = preprocessor.getTokenReferenceSize();

        var il = InputLayer.build(preprocessor.getInputShape());
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
                .train(preprocessor, 1, 0.05);

        String output = model.generateOutput("For there are scoffers who maintain", 1000);
        System.out.println(output);
    }
}
