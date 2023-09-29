import checkpoint.CheckpointManager;
import layers.DenseLayer;
import layers.InputLayer;
import layers.SoftmaxLayer;
import models.LanguageModel;
import preprocessing.TrainingDataPreprocessor;
import preprocessing.datasources.TextFileReader;
import preprocessing.tokenization.CharTokenizer;
import preprocessing.vectorization.OneHotVectorizer;
import telemetry.TrainingMonitor;
import util.Activations;
import util.LossFunctions;

public class Training {
  public static void main(String[] args) {
    var dir = "/home/jonas/code/nlp-from-scratch/";

    int inputSize = 40;
    var reader = new TextFileReader(dir + "assets/odyssey_short.txt", 1);
    var preprocessor = new TrainingDataPreprocessor(reader, CharTokenizer.build(), OneHotVectorizer.build(), inputSize, 1, 20);
    var model = new LanguageModel();
    var trainingMonitor = new TrainingMonitor(dir + "metrics");
    var checkpointManager = new CheckpointManager(dir + "checkpoints");

    var tokenRefSize = preprocessor.getTokenReferenceSize();

    var il = InputLayer.build(preprocessor.getInputShape());
    var dl0 = DenseLayer.build(inputSize * tokenRefSize).setActivationFn(Activations.TANH);
    var dl1 = DenseLayer.build(inputSize * tokenRefSize).setActivationFn(Activations.TANH);
    var dl2 = DenseLayer.build(tokenRefSize).setActivationFn(Activations.SOFTPLUS);
    var sml = SoftmaxLayer.build(tokenRefSize);

    model
      .setName("basic-lm")
      .setPreprocessor(preprocessor)
      .addLayer(il)
      .addLayer(dl0)
//      .addLayer(dl1)
      .addLayer(dl2)
      .addLayer(sml)
      .setLossFunction(LossFunctions.CATEGORICAL_CROSSENTROPY)
      .initialize()
      .attachTelemetry(trainingMonitor)
      .attachCheckpointManager(checkpointManager)
      .train(1, 0.04f) // TODO: Fix this. There should be no need for setting preproc twice
      .createCheckpoint()
      .commitMetrics();

    String output = model.generateOutput("for there are scoffers who maintain", 1000);
    System.out.printf("\nOutput:\n%s\n", output);
  }
}
