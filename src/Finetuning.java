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
    var dir = "/home/jonas/code/nlp-from-scratch/";
    var modelReader = new ModelReader<LanguageModel>();

    var reader = new TextFileReader(dir + "assets/odyssey.txt", 100);
    var preprocessor = new TrainingDataPreprocessor(reader, CharTokenizer.build(), OneHotVectorizer.build(), 3, 64);
    var trainingMonitor = new TrainingMonitor(dir + "metrics");
    var checkpointManager = new CheckpointManager(dir + "checkpoints");

    // TODO: Add tokenReference compatibility check

    LanguageModel model = modelReader
//      .read(dir + "checkpoints/basic-lm.1696619402443.model");
      .read(dir + "checkpoints/basic-lm.1696621142022.model");

//  Commit in case of SIGTERM
    Runtime.getRuntime().addShutdownHook(new Thread(() -> model
      .commitCheckpoint()
      .commitMetrics()));

    model
      .attachTelemetry(trainingMonitor)
      .attachCheckpointManager(checkpointManager)
      .setPreprocessor(preprocessor)
      .initialize()
      .train(1, 0.0003f);
//      .commitCheckpoint()
//      .commitMetrics();
  }
}
