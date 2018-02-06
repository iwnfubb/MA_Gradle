package utils;

import java.io.*;
import java.util.ArrayList;

public class CSVReaderHistogram extends AbstractCSVReader<Double> {

    public CSVReaderHistogram(File file, String csvSplitBy) {
        super(file, csvSplitBy);
    }

    @Override
    public void read() {
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

    @Override
    public ArrayList<Double> getValues() {
        return list;
    }
}
