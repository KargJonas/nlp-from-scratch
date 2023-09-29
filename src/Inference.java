import models.LanguageModel;
import preprocessing.PromptPreprocessor;
import preprocessing.TrainingDataPreprocessor;
import preprocessing.datasources.StringReader;
import preprocessing.tokenization.CharTokenizer;
import preprocessing.vectorization.OneHotVectorizer;
import util.ModelReader;

public class Inference {
  public static void main(String[] args) {
    var dir = "/home/jonas/code/nlp-from-scratch/";

    int inputSize = 40;
    var pp = new TrainingDataPreprocessor(new StringReader(" ", 1), CharTokenizer.build(), OneHotVectorizer.build(), inputSize, 1, 20);
    var preprocessor = new PromptPreprocessor(pp);
    var importedModel = ModelReader.read(dir + "checkpoints/basic-lm.280923-221030.model");
    var importedLM = new LanguageModel(importedModel);

    LanguageModel model1 = (LanguageModel) importedLM
      .initialize();

    String output1 = model1.generateOutput("for there are scoffers who maintain", 1000);
    System.out.printf("\nOutput:\n%s\n", output1);

    // TODO: Current problem: Integrate the preprocessor into the checkpoint for use with PromptPreprocessor during inference
    //  OR find a nicer solution to this problem
  }
}
