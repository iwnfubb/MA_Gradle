package utils;

import java.io.BufferedReader;
import java.io.File;
import java.util.ArrayList;

public class AbstractCSVReader<T> {
    BufferedReader br = null;
    String line = "";
    String csvSplitBy = ";";
    ArrayList<T> list;
    File file;

    public AbstractCSVReader(File file, String csvSplitBy) {
        this.file = file;
        this.csvSplitBy = csvSplitBy;
    }

    public void read() {
    }

    public ArrayList<T> getValues() {
        return list;
    }
}
