import checkpoint.CheckpointManager;
import models.LanguageModel;
import preprocessing.TrainingDataPreprocessor;
import preprocessing.datasources.TextFileReader;
import preprocessing.tokenization.CharTokenizer;
import preprocessing.vectorization.OneHotVectorizer;
import telemetry.TrainingMonitor;
import util.ModelReader;

public class Finetuning {
  public static void main(String[] args) {
    int inputSize = 40;

    var dir = "/home/jonas/code/nlp-from-scratch/";
    var modelReader = new ModelReader<LanguageModel>();

    var reader = new TextFileReader(dir + "assets/odyssey_short.txt", 1);
    var preprocessor = new TrainingDataPreprocessor(reader, CharTokenizer.build(), OneHotVectorizer.build(), inputSize, 1, 20);
    var trainingMonitor = new TrainingMonitor(dir + "metrics");
    var checkpointManager = new CheckpointManager(dir + "checkpoints");

    // TODO: Add tokenReference compatibility check

    var importedModel = modelReader
      .read(dir + "checkpoints/basic-lm.300923-153900.model")
      .attachTelemetry(trainingMonitor)
      .attachCheckpointManager(checkpointManager)
      .setPreprocessor(preprocessor)
      .initialize()
      .train(10, 0.0003f)
      .createCheckpoint()
      .commitMetrics();
  }
}
