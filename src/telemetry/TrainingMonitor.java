package telemetry;

import util.DirectoryHandler;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

public class TrainingMonitor extends DirectoryHandler {

  ArrayList<Float> metric = new ArrayList<>();

  public TrainingMonitor(String outDirectory) {
    super(outDirectory);
  }

  public void add(float loss) {
    metric.add(loss);
  }

  public void commit() {
    String fileName = getFileName("csv");
    File file = new File(fileName);
    PrintWriter writer;

    if (file.exists()) {
      throw new RuntimeException("Cannot create checkpoint, file exists.");
    }

    try {
      writer = new PrintWriter(new FileWriter(file, true));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    for (float dataPoint : metric) {
      writer.println(dataPoint);
    }

    System.out.printf("Successfully committed training metrics: \"%s\"\n", fileName);

    writer.close();
  }
}
