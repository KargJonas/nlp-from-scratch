package preprocessing.datasources;

import java.io.*;
import java.nio.file.Files;
import java.util.Iterator;

/**
 * Reads text files section-by-section.
 * Overflowing chars from the end of the files may be omitted to guarantee the section size to be sectionSize.
 */
public class TextFileReader implements TextSource {

  String path;
  int sectionSize;

  /**
   * Instantiates a Datasources.TextFileReader
   * @param path Path that references a text file
   * @param sectionSize Size of the returned sections.
   */
  public TextFileReader(String path, int sectionSize) {
    this.path = path;
    this.sectionSize = sectionSize;

    File file = new File(path);
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

  @Override
  public Iterator<String> iterator() {

    return new Iterator<>() {
      final BufferedReader reader;
      final char[] buffer = new char[sectionSize];
      int state;

      {
        try {
          reader = new BufferedReader(new FileReader(path));
        } catch (FileNotFoundException e) {
          throw new RuntimeException(String.format("File not found: %s", path));
        }

        // Populate buffer and set state
        readIntoBuffer();
      }

      private void readIntoBuffer() {
        try {
          state = reader.read(buffer, 0, sectionSize);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }

      @Override
      public boolean hasNext() {
        // There is a next element if the
        return state != -1 && state >= buffer.length;
      }

      @Override
      public String next() {
        String section = new String(buffer);
        readIntoBuffer();
        return section;
      }
    };
  }
}
