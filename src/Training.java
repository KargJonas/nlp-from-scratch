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

    var reader = new TextFileReader(dir + "assets/abc.txt", 20);
    var preprocessor = new TrainingDataPreprocessor(reader, CharTokenizer.build(), OneHotVectorizer.build(), 5, 10);
    var model = new LanguageModel();
    var trainingMonitor = new TrainingMonitor(dir + "metrics");
    var checkpointManager = new CheckpointManager(dir + "checkpoints");

    var tokenRefSize = preprocessor.getTokenReferenceSize();

    // Commit in case of SIGTERM
    Runtime.getRuntime().addShutdownHook(new Thread(() -> model
      .commitCheckpoint()
      .commitMetrics()));

    model
      .setName("basic-lm")
      .setPreprocessor(preprocessor)
      .addLayer(InputLayer.build(preprocessor.getInputSize()))
      .addLayer(DenseLayer    .build(200)     .setActivationFn(Activations.TANH))
      .addLayer(RecurrentLayer.build(200)     .setActivationFn(Activations.TANH).setActivationInitializer(Initializers.ZEROS()))
//      .addLayer(RecurrentLayer.build(300)     .setActivationFn(Activations.TANH).setActivationInitializer(Initializers.ZEROS()))
//      .addLayer(RecurrentLayer.build(300)     .setActivationFn(Activations.TANH).setActivationInitializer(Initializers.ZEROS()))
      .addLayer(DenseLayer    .build(100)     .setActivationFn(Activations.TANH))
      .addLayer(DenseLayer    .build(tokenRefSize).setActivationFn(Activations.IDENTITY))
      .addLayer(SoftmaxLayer  .build(tokenRefSize))
      .setLossFunction(LossFunctions.CATEGORICAL_CROSSENTROPY)
      .initialize()
      .attachTelemetry(trainingMonitor)
      .attachCheckpointManager(checkpointManager)
      .train(1, 0.01f);
//      .commitCheckpoint()
//      .commitMetrics();

//    String output = model.generateOutput("for there are scoffers who maintain", 1000);
    String output = model.generateOutput("aaaa", 1000);
    System.out.printf("\nOutput:\n%s\n", output);
  }
}
