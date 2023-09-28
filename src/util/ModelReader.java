package util;

import models.Model;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Files;

public class ModelReader {

  public static Model read(String path) {

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

    if (file.isDirectory() || type == null) {
      throw new RuntimeException("The provided path references an invalid object.");
    }

    Model model = null;

    try {
      FileInputStream fileIn = new FileInputStream(file);
      ObjectInputStream in = new ObjectInputStream(fileIn);

      model = (Model) in.readObject();
    } catch (Exception e) {
      System.out.println(e);
    }

    if (model == null) {
      System.out.println("Failed to read model");
    }

    return model;
  }
}
