package guicontroler;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
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
            double[] v1 = new double[values1.size()];
            double[] v2 = new double[values1.size()];
            double[] frame = new double[values1.size()];
            int counter = 0;
            for (int i = 0; i < values1.size(); i++) {
                v1[i] = values1.get(i).getPosture_in_double();
                v2[i] = values2.get(i).getPosture_in_double();
                if (v1[i] == v2[i]) {
                    counter++;
                }
                frame[i] = i + 1;
            }
            String titel = ("True: " + counter + " from: " + values1.size() + " Quote: " + (double) counter / (double) values1.size());
            XYChart chart = QuickChart.getChart(titel, "Frame", "Posture", "real", frame, v1);
            chart.addSeries("test", frame, v2);
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
