package Data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class PlainTextReader {
    String path;
    Integer currentLine = 0;
    BufferedReader file;

    public PlainTextReader(String path) {

        // Instantiate bufferedReader
        try {
            file = new BufferedReader(new FileReader(path));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(
                    String.format("File not found: %s\n", path));
        }
    }

    public String readLine() {
        currentLine++;

        try {
            return file.readLine();
        }  catch (IOException e) {
            throw new RuntimeException(String.format(
                    "Issue while reading line %s of file: %s\n", currentLine, path));
        }
    }
}
