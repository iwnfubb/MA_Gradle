package utils;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;

public class AverageHistogram {
    public ArrayList<Double> vertical_standing;
    public ArrayList<Double> vertical_bending;
    public ArrayList<Double> vertical_laying;
    public ArrayList<Double> vertical_sitting;
    public ArrayList<Double> horizontal_standing;
    public ArrayList<Double> horizontal_bending;
    public ArrayList<Double> horizontal_laying;
    public ArrayList<Double> horizontal_sitting;

    public AverageHistogram() {
        iniVertical_standing();
        iniVertical_bending();
        iniVertical_laying();
        iniVertical_sitting();

        iniHorizontal_standing();
        iniHorizontal_bending();
        iniHorizontal_laying();
        iniHorizontal_sitting();
    }

    private void iniVertical_standing() {
        URL url = getClass().getResource("../../resources/" + "v_standing.csv");
        File file = new File(url.getPath());
        CSVReaderHistogram reader = new CSVReaderHistogram(file, ";");
        reader.read();
        vertical_standing = reader.getValues();
    }

    private void iniVertical_bending() {
        URL url = getClass().getResource("../../resources/" + "v_bending.csv");
        File file = new File(url.getPath());
        CSVReaderHistogram reader = new CSVReaderHistogram(file, ";");
        reader.read();
        vertical_bending = reader.getValues();
    }

    private void iniVertical_laying() {
        URL url = getClass().getResource("../../resources/" + "v_laying.csv");
        File file = new File(url.getPath());
        CSVReaderHistogram reader = new CSVReaderHistogram(file, ";");
        reader.read();
        vertical_laying = reader.getValues();
    }

    private void iniVertical_sitting() {
        URL url = getClass().getResource("../../resources/" + "v_sitting.csv");
        File file = new File(url.getPath());
        CSVReaderHistogram reader = new CSVReaderHistogram(file, ";");
        reader.read();
        vertical_sitting = reader.getValues();
    }

    private void iniHorizontal_standing() {
        URL url = getClass().getResource("../../resources/" + "h_standing.csv");
        File file = new File(url.getPath());
        CSVReaderHistogram reader = new CSVReaderHistogram(file, ";");
        reader.read();
        horizontal_standing = reader.getValues();
    }

    private void iniHorizontal_bending() {
        URL url = getClass().getResource("../../resources/" + "h_bending.csv");
        File file = new File(url.getPath());
        CSVReaderHistogram reader = new CSVReaderHistogram(file, ";");
        reader.read();
        horizontal_bending = reader.getValues();
    }

    private void iniHorizontal_laying() {
        URL url = getClass().getResource("../../resources/" + "h_laying.csv");
        File file = new File(url.getPath());
        CSVReaderHistogram reader = new CSVReaderHistogram(file, ";");
        reader.read();
        horizontal_laying = reader.getValues();
    }

    private void iniHorizontal_sitting() {
        URL url = getClass().getResource("../../resources/" + "h_sitting.csv");
        File file = new File(url.getPath());
        CSVReaderHistogram reader = new CSVReaderHistogram(file, ";");
        reader.read();
        horizontal_sitting = reader.getValues();
    }

    public static void main(String[] args) {
        AverageHistogram test = new AverageHistogram();
        System.out.println(test.vertical_bending.size());
        System.out.println(test.vertical_standing.size());
        System.out.println(test.vertical_laying.size());
        System.out.println(test.vertical_sitting.size());

        System.out.println(test.horizontal_sitting.size());
        System.out.println(test.horizontal_laying.size());
        System.out.println(test.horizontal_bending.size());
        System.out.println(test.horizontal_standing.size());
    }
}
