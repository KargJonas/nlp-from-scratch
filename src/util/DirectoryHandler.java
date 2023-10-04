package util;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

public class DirectoryHandler {

  protected static final SimpleDateFormat sdf = new SimpleDateFormat("ddMMyy-HHmmss");
  protected String name;
  long dateTime = System.currentTimeMillis();

  protected final String outDirectory;
  protected final File directory;

  public DirectoryHandler(String outDirectory) {
    this.outDirectory = outDirectory;
    directory = new File(outDirectory);

    if (!directory.exists()) {
      throw new RuntimeException("The provided directory does not exist.");
    }

    if (!directory.isDirectory()) {
      throw new RuntimeException("The provided path references a non-directory object.");
    }
  }

  public String getFileName(String fileExtension) {
    return name == null
      ? String.format("%s/%s.%s", outDirectory, dateTime, fileExtension)
      : String.format("%s/%s.%s.%s", outDirectory, name, dateTime, fileExtension);
  }

  public void setName(String name) {
    this.name = name;
  }
}
