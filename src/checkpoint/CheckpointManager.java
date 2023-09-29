package checkpoint;

import models.Model;
import util.DirectoryHandler;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

public class CheckpointManager extends DirectoryHandler {

  ObjectOutputStream objectOutputStream;

  public CheckpointManager(String outDirectory) {
    super(outDirectory);
  }

  public void createCheckpoint(Model model) {
    String fileName = String.format("%s/%s.%s.model", outDirectory, model.getName(), dateTime);

    try {
      FileOutputStream fileOutputStream = new FileOutputStream(fileName);
      objectOutputStream = new ObjectOutputStream(fileOutputStream);
      objectOutputStream.writeObject(model);
      System.out.printf("Successfully created checkpoint: \"%s\"\n", fileName);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
