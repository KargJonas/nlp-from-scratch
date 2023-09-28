package util;

import java.io.File;

public class FsHandler {

  protected String outDirectory;
  protected File directory;

  public FsHandler(String outDirectory) {
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
