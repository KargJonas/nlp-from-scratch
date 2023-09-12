package preprocessing.datasources;

import java.util.Iterator;

public class StringReader implements TextSource {

  String text;
  int sectionSize;

  public StringReader(String text, int sectionSize) {
    this.text = text;
    this.sectionSize = sectionSize;
  }

  @Override
  public Iterator<String> iterator() {
    return new Iterator<>() {
      int index = 0;

      @Override
      public boolean hasNext() {
        return index + sectionSize <= text.length();
      }

      @Override
      public String next() {
        String section = text.substring(index, index + sectionSize);
        index += sectionSize;
        return section;
      }
    };
  }
}
