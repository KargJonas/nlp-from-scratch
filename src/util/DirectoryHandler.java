package util;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

public class DirectoryHandler {

  protected static final SimpleDateFormat sdf = new SimpleDateFormat("ddMMyy-HHmmss");
  private final Date now = new Date();
  protected final String dateTime = sdf.format(now);

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
}
