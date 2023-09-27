package telemetry;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Date;

public class TrainingMonitor {

  static final SimpleDateFormat sdf = new SimpleDateFormat("ddMMyy-HHmmss");
  final Date now = new Date();

  ArrayList<Double> metric = new ArrayList<>();

  String outDirectory;
  File directory;

  public TrainingMonitor(String outDirectory) {
    this.outDirectory = outDirectory;
    directory = new File(outDirectory);

//    String type;
//
//    try {
//      type = Files.probeContentType(directory.toPath());
//    } catch (IOException e) {
//      throw new RuntimeException(e);
//    }

    if (!directory.exists()) {
      throw new RuntimeException("The provided directory does not exist.");
    }

    if (!directory.isDirectory()) {
      throw new RuntimeException("The provided path references a non-directory object.");
    }
  }

  public void add(double loss) {
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

    for (Double dataPoint : metric) {
      writer.println(dataPoint);
    }

    writer.close();
  }
}
