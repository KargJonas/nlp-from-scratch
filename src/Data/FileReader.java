package Data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.Buffer;
import java.util.Iterator;

public class FileReader implements Iterable<String> {
    String path;

    public FileReader(String path) {
        this.path = path;
    }

    private String readLine(BufferedReader reader) {
        try {
            return reader.readLine();
        }  catch (IOException e) {
            return null;
        }
    }

    @Override
    public Iterator<String> iterator() {
        BufferedReader reader;

        try {
            reader = new BufferedReader(new java.io.FileReader(path));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(
              String.format("File not found: %s\n", path));
        }

        return new Iterator<>() {
            String line = readLine(reader);

            @Override
            public boolean hasNext() {
                return line != null;
            }

            @Override
            public String next() {
                String currentLine = line;
                line = readLine(reader);
                return currentLine;
            }
        };
    }
}
