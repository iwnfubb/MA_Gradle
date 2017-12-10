package utils;

import java.io.*;
import java.net.URI;
import java.net.URL;
import java.util.ArrayList;

public class CVSReader {
    String csvFile = "";
    BufferedReader br = null;
    String line = "";
    String csvSplitBy = ";";
    ArrayList<Double> list;

    public CVSReader(String csvFile, String csvSplitBy) {
        this.csvFile = csvFile;
        this.csvSplitBy = csvSplitBy;
    }


    public void read() {
        URL url = getClass().getResource("../../resources/" + csvFile);
        File file = new File(url.getPath());
        list = new ArrayList<>();
        try {
            br = new BufferedReader(new FileReader(file));
            while ((line = br.readLine()) != null) {
                String[] row = line.split(csvSplitBy);
                list.add(Double.parseDouble(row[1]));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public ArrayList<Double> getValues() {
        return list;
    }

}
