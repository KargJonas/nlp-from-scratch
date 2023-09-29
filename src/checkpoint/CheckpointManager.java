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

  public <T extends Model>void createCheckpoint(T model) {
    String fileName = String.format("%s/%s.%s.model", outDirectory, model.name, dateTime);

    try {
      FileOutputStream fileOutputStream = new FileOutputStream(fileName);
      objectOutputStream = new ObjectOutputStream(fileOutputStream);
      objectOutputStream.writeObject(model);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
