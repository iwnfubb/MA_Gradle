package utils;

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
        CVSReader reader = new CVSReader("v_standing.csv", ";");
        reader.read();
        vertical_standing = reader.getValues();
    }

    private void iniVertical_bending() {
        CVSReader reader = new CVSReader("v_bending.csv", ";");
        reader.read();
        vertical_bending = reader.getValues();
    }

    private void iniVertical_laying() {
        CVSReader reader = new CVSReader("v_laying.csv", ";");
        reader.read();
        vertical_laying = reader.getValues();
    }

    private void iniVertical_sitting() {
        CVSReader reader = new CVSReader("v_sitting.csv", ";");
        reader.read();
        vertical_sitting = reader.getValues();
    }

    private void iniHorizontal_standing() {
        CVSReader reader = new CVSReader("h_standing.csv", ";");
        reader.read();
        horizontal_standing = reader.getValues();
    }

    private void iniHorizontal_bending() {
        CVSReader reader = new CVSReader("h_bending.csv", ";");
        reader.read();
        horizontal_bending = reader.getValues();
    }

    private void iniHorizontal_laying() {
        CVSReader reader = new CVSReader("h_laying.csv", ";");
        reader.read();
        horizontal_laying = reader.getValues();
    }

    private void iniHorizontal_sitting() {
        CVSReader reader = new CVSReader("h_sitting.csv", ";");
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
