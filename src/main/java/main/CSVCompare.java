package main;

import guicontroler.GUIController_CSVCompare;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.GridPane;
import javafx.stage.Stage;
import org.opencv.core.Core;

public class CSVCompare extends Application {
    @Override
    public void start(Stage primaryStage) throws Exception {
        try {
            FXMLLoader root = new FXMLLoader(getClass().getResource("../../resources/csvcompare.fxml"));
            GridPane rootElement = root.load();

            primaryStage.setTitle("Compare");
            primaryStage.setScene(new Scene(rootElement, 800, 100));
            primaryStage.setResizable(false);
            primaryStage.show();

            GUIController_CSVCompare controller = root.getController();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        launch(args);
    }


    /*
    public static void main(String[] args) {
        int numCharts = 5;

        List<XYChart> charts = new ArrayList<XYChart>();

        for (int i = 0; i < numCharts; i++) {
            XYChart chart = new XYChartBuilder().xAxisTitle("X").yAxisTitle("Y").width(600).height(400).build();
            chart.getStyler().setYAxisMin(-10.0);
            chart.getStyler().setYAxisMax(10.0);
            XYSeries series = chart.addSeries("" + i, null, getRandomWalk(200));
            series.setMarker(SeriesMarkers.NONE);
            charts.add(chart);
        }
        new SwingWrapper<XYChart>(charts).displayChartMatrix();
    }

    /**
     * Generates a set of random walk data
     *
     * @param numPoints
     * @return
     */
    private static double[] getRandomWalk(int numPoints) {

        double[] y = new double[numPoints];
        y[0] = 0;
        for (int i = 1; i < y.length; i++) {
            y[i] = y[i - 1] + Math.random() - .5;
        }
        return y;
    }

}
