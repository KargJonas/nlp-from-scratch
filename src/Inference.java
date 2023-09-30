import models.LanguageModel;
import util.ModelReader;

public class Inference {
  public static void main(String[] args) {
    var dir = "/home/jonas/code/nlp-from-scratch/";
    var modelReader = new ModelReader<LanguageModel>();

    var importedModel = modelReader
      .read(dir + "checkpoints/basic-lm.300923-182904.model")
      .initialize();

    String output1 = importedModel.generateOutput("then eurymachus son of polybus answered", 1000);
    System.out.printf("\nOutput:\n%s\n", output1);
  }
}
