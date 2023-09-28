package telemetry;

import util.FsHandler;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Date;

public class TrainingMonitor extends FsHandler {

  static final SimpleDateFormat sdf = new SimpleDateFormat("ddMMyy-HHmmss");
  final Date now = new Date();
  ArrayList<Float> metric = new ArrayList<>();

  public TrainingMonitor(String outDirectory) {
    super(outDirectory);
  }

  public void add(float loss) {
    metric.add(loss);
  }

  public void commit() {
    String fileName = String.format("%s/%s.csv", outDirectory, sdf.format(now));
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
