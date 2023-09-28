package util;

import java.io.File;

public class DirectoryHandler {

  protected String outDirectory;
  protected File directory;

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
