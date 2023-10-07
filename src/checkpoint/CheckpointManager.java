package checkpoint;

import models.Model;
import util.DirectoryHandler;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

public class CheckpointManager extends DirectoryHandler {

  ObjectOutputStream objectOutputStream;

  public CheckpointManager(String outDirectory) {
    super(outDirectory);
  }

  public void commitCheckpoint(Model model) {
    String fileName = getFileName("model");
    File file = new File(fileName);

    if (file.exists()) {
      throw new RuntimeException("Cannot create checkpoint, file exists.");
    }

    try {
      FileOutputStream fileOutputStream = new FileOutputStream(file);
      objectOutputStream = new ObjectOutputStream(fileOutputStream);
      objectOutputStream.writeObject(model);
      System.out.printf("Successfully created checkpoint: \"%s\"\n", fileName);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
