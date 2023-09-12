package telemetry;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class TrainingMonitor {

  File file;

  public TrainingMonitor(String outPath) {
    file = new File(outPath);

    String type;

    try {
      type = Files.probeContentType(file.toPath());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    if (!file.exists()) {
      throw new RuntimeException("The provided file does not exist.");
    }

    if (file.isDirectory() || type == null || !type.startsWith("text")) {
      throw new RuntimeException("The provided path references an invalid file.");
    }
  }
}
