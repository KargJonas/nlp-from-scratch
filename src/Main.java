import layers.DenseLayer;
import layers.InputLayer;
import layers.SoftmaxLayer;
import models.LanguageModel;
import preprocessing.TrainingDataPreprocessor;
import telemetry.TrainingMonitor;
import util.Activations;
import preprocessing.datasources.TextFileReader;
import preprocessing.tokenization.CharTokenizer;
import preprocessing.vectorization.OneHotVectorizer;
import util.LossFunctions;

public class Main {
    public static void main(String[] args) {
        int inputSize = 40;

        var dir = "/home/jonas/code/nlp-from-scratch/";
        var reader       = new TextFileReader(dir + "assets/odyssey_short.txt", 1);
        var preprocessor = new TrainingDataPreprocessor(reader, CharTokenizer.build(), OneHotVectorizer.build(), inputSize, 1, 20);
        var model        = new LanguageModel(preprocessor);
        var trainingMonitor = new TrainingMonitor(dir + "metrics");

        var tokenRefSize = preprocessor.getTokenReferenceSize();

        var il = InputLayer.build(preprocessor.getInputShape());
        var dl0 = DenseLayer.build(inputSize * tokenRefSize).setActivationFn(Activations.TANH);
        var dl1 = DenseLayer.build(inputSize * tokenRefSize).setActivationFn(Activations.TANH);
        var dl2 = DenseLayer.build(tokenRefSize).setActivationFn(Activations.SOFTPLUS);
        var sml = SoftmaxLayer.build(tokenRefSize);

        model
                .addLayer(il)
                .addLayer(dl0)
//                .addLayer(dl1)
                .addLayer(dl2)
                .addLayer(sml)
                .setLossFunction(LossFunctions.CATEGORICAL_CROSSENTROPY)
                .initialize()
                .attachTelemetry(trainingMonitor)
                .train(preprocessor, 2, 0.1f);

        String output = model.generateOutput("for there are scoffers who maintain", 1000);
        System.out.printf("\nOutput START\n%s\nOutput END", output);
    }
}
