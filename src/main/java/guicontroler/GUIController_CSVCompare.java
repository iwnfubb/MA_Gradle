package guicontroler;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.markers.SeriesMarkers;
import utils.CSVReaderEvaluation;
import utils.EvaluationValue;
import utils.Utils;

import java.io.File;
import java.util.ArrayList;

public class GUIController_CSVCompare {
    private String fileName1;
    private String fileName2;
    FileChooser fileChooser = new FileChooser();
    File file1;
    File file2;
    int skip = 250;
    @FXML
    private Label path1;
    @FXML
    private Label path2;


    @FXML
    protected void compare() {

        if (file1 != null && file2 != null) {
            CSVReaderEvaluation reader1 = new CSVReaderEvaluation(file1, ",");
            reader1.read();
            ArrayList<EvaluationValue> values1 = reader1.getValues();
            CSVReaderEvaluation reader2 = new CSVReaderEvaluation(file2, ",");
            reader2.read();
            ArrayList<EvaluationValue> values2 = reader2.getValues();

            if (values1.size() != values2.size()) {
                Alert alert = new Alert(Alert.AlertType.ERROR);
                alert.setTitle("Error");
                alert.setHeaderText("File size doesn't match");
                alert.setContentText("Size 1: " + values1.size() + " Size 2: " + values2.size());
                alert.showAndWait();
                return;
            }

            //value1.size() == value2.size()
            double[] v1 = new double[values1.size() - skip];
            double[] v2 = new double[values1.size() - skip];
            double[] frame = new double[values1.size() - skip];
            int counter = 0;
            for (int i = skip; i < values1.size(); i++) {
                v1[i - skip] = values1.get(i).getPosture_in_double();
                v2[i - skip] = values2.get(i).getPosture_in_double();
                if (v1[i - skip] == v2[i - skip]) {
                    counter++;
                }
                frame[i-skip] = i - skip + 1;
            }

            // Create Chart
            XYChart chart = new XYChartBuilder().width(800).height(600).title(getClass().getSimpleName()).xAxisTitle("Frame").yAxisTitle("Posture").build();

            // Customize Chart
            chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
            chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
            chart.getStyler().setYAxisLabelAlignment(Styler.TextAlignment.Right);
            chart.getStyler().setYAxisDecimalPattern("#,###.##");
            chart.getStyler().setPlotMargin(0);
            chart.getStyler().setPlotContentSize(.95);

            String titel = ("True: " + counter + " from: " + (values1.size() - skip) + " Quote: " + (double) counter / (double) (values1.size() - skip));
            chart.setTitle(titel);
            XYSeries real = chart.addSeries("real", frame, v1);
            real.setMarker(SeriesMarkers.NONE);
            XYSeries test = chart.addSeries("test", frame, v2);
            test.setMarker(SeriesMarkers.NONE);

            // Show it
            new SwingWrapper(chart).displayChart();
        }
    }


    private void updateImageView(ImageView view, Image image) {
        if (image != null) {
            Platform.runLater(() -> {
                view.setFitWidth(image.getWidth());
                view.setFitHeight(image.getHeight());
            });

            Utils.onFXThread(view.imageProperty(), image);
        }
    }

    @FXML
    protected void openCSVData1() {
        //Set extension filter
        FileChooser.ExtensionFilter extFilter = new FileChooser.ExtensionFilter("Video files (*.csv)", "*.csv");
        fileChooser.getExtensionFilters().add(extFilter);
        //Show open file dialog
        file1 = fileChooser.showOpenDialog(null);
        if (file1 != null) {
            Platform.runLater(() -> {
                path1.setText(file1.getName());
            });
        }
    }

    @FXML
    protected void openCSVData2() {
        //Set extension filter
        FileChooser.ExtensionFilter extFilter = new FileChooser.ExtensionFilter("Video files (*.csv)", "*.csv");
        fileChooser.getExtensionFilters().add(extFilter);
        //Show open file dialog
        file2 = fileChooser.showOpenDialog(null);
        if (file2 != null) {
            Platform.runLater(() -> {
                path2.setText(file2.getName());
            });
        }
    }
}
