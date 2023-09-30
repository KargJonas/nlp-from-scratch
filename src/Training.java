import checkpoint.CheckpointManager;
import layers.DenseLayer;
import layers.InputLayer;
import layers.RecurrentLayer;
import layers.SoftmaxLayer;
import models.LanguageModel;
import preprocessing.TrainingDataPreprocessor;
import preprocessing.datasources.TextFileReader;
import preprocessing.tokenization.CharTokenizer;
import preprocessing.vectorization.OneHotVectorizer;
import telemetry.TrainingMonitor;
import util.Activations;
import util.Initializers;
import util.LossFunctions;

public class Training {
  public static void main(String[] args) {
    var dir = "/home/jonas/code/nlp-from-scratch/";

    var reader = new TextFileReader(dir + "assets/abc.txt", 100);
    var preprocessor = new TrainingDataPreprocessor(reader, CharTokenizer.build(), OneHotVectorizer.build(), 1, 20);
    var model = new LanguageModel();
    var trainingMonitor = new TrainingMonitor(dir + "metrics");
    var checkpointManager = new CheckpointManager(dir + "checkpoints");

    var tokenRefSize = preprocessor.getTokenReferenceSize();

    model
      .setName("basic-lm")
      .setPreprocessor(preprocessor)
      .addLayer(InputLayer    .build(preprocessor.getInputShape()))
      .addLayer(DenseLayer    .build(128)     .setActivationFn(Activations.TANH))
      .addLayer(RecurrentLayer.build(128)     .setActivationFn(Activations.RELU).setActivationInitializer(Initializers.ZEROS()))
      .addLayer(DenseLayer    .build(128)     .setActivationFn(Activations.RELU))
      .addLayer(DenseLayer    .build(tokenRefSize).setActivationFn(Activations.SOFTPLUS))
      .addLayer(SoftmaxLayer  .build(tokenRefSize))
      .setLossFunction(LossFunctions.CATEGORICAL_CROSSENTROPY)
      .initialize()
      .attachTelemetry(trainingMonitor)
      .attachCheckpointManager(checkpointManager)
      .train(10, 0.1f)
      .createCheckpoint()
      .commitMetrics();

//    String output = model.generateOutput("for there are scoffers who maintain", 1000);
    String output = model.generateOutput("db", 1000);
    System.out.printf("\nOutput:\n%s\n", output);
  }
}
