import models.LanguageModel;
import util.ModelReader;

public class Inference {
  public static void main(String[] args) {
    var dir = "/home/jonas/code/nlp-from-scratch/";
    var modelReader = new ModelReader<LanguageModel>();

    var importedModel = modelReader
      .read(dir + "checkpoints/basic-lm.1696669025739.model")
      .initialize();

    String output1 = importedModel.generateOutput("Ich hab ein boot.", 1000);
    System.out.printf("\nOutput:\n%s\n", output1);
  }
}
