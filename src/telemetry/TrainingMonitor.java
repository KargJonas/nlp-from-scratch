package telemetry;

import util.DirectoryHandler;

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
    String fileName = String.format("%s/%s.csv", outDirectory, dateTime);
    PrintWriter writer;

    try {
      writer = new PrintWriter(new FileWriter(fileName, true));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    for (float dataPoint : metric) {
      writer.println(dataPoint);
    }

    writer.close();
  }
}
