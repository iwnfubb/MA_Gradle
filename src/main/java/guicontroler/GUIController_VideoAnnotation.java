package guicontroler;

import imageprocess.ImageProcess_ObjectTracking;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.stage.FileChooser;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import utils.CSVWriter;
import utils.EvaluationValue;
import utils.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class GUIController_VideoAnnotation {
    @FXML
    private Label frame_number;
    @FXML
    private Label bb_values;
    @FXML
    private CheckBox person_there;
    @FXML
    private RadioButton standing;
    @FXML
    private RadioButton bending;
    @FXML
    private RadioButton sitting;
    @FXML
    private RadioButton laying;

    @FXML
    private RadioButton moving;
    @FXML
    private RadioButton not_moving;

    @FXML
    private RadioButton ok;
    @FXML
    private RadioButton not_ok;


    @FXML
    private Button openButton;

    @FXML
    private Slider timerbar;


    @FXML
    private ImageView imageView;

    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that realizes the video capture
    private VideoCapture capture = new VideoCapture();
    ImageProcess_ObjectTracking imgProcess = new ImageProcess_ObjectTracking(this.capture);
    // a flag to change the button behavior
    private boolean cameraActive = false;
    FileWriter writer;

    private String fileName;
    FileChooser fileChooser = new FileChooser();
    File file;
    final ToggleGroup posture = new ToggleGroup();
    final ToggleGroup isMoving = new ToggleGroup();
    final ToggleGroup status = new ToggleGroup();

    double startX = 0;
    double startY = 0;
    double endX = 0;
    double endY = 0;
    Mat originalFrame = new Mat();
    ArrayList<EvaluationValue> list = new ArrayList<>();

    /**
     * The action triggered by pushing the button on the GUI
     *
     * @param event the push button event
     */
    @FXML
    protected void openVideo(ActionEvent event) {

        if (!this.cameraActive) {
            // start the video capture
            ini();
            if (file != null) {
                fileName = file.getPath();
                this.capture.open(fileName);
                // is the video stream available?
                if (this.capture.isOpened()) {
                    iniTimerBar();
                    resetAllValue();
                    this.cameraActive = true;
                    imgProcess.getOriginalFrame().copyTo(originalFrame);
                    drawRect(originalFrame);
                    Platform.runLater(() -> frame_number.setText(capture.get(Videoio.CAP_PROP_POS_FRAMES) + ""));
                    this.timer = Executors.newSingleThreadScheduledExecutor();
                    createCVSFile();
                } else {
                    // log the error
                    System.err.println("Impossible to open the camera connection...");
                }
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;

            // stop the timer
            //this.stopAcquisition();
        }
    }

    private void drawRect(Mat mat) {
        if (!mat.empty()) {
            Mat newMat = new Mat();
            mat.copyTo(newMat);
            if (person_there.isSelected()) {
                Imgproc.rectangle(
                        newMat,
                        new Point(startX, startY),
                        new Point(endX, endY),
                        new Scalar(0, 255, 0), 5);
                setTextOnBBLabel();
            }
            Image image = Utils.mat2Image(newMat);
            updateImageView(imageView, image);
        } else {
            System.err.print("End of Stream");
        }
    }

    @FXML
    protected void nextFrame(ActionEvent event) {
        if (cameraActive) {
            Platform.runLater(() -> frame_number.setText(capture.get(Videoio.CAP_PROP_POS_FRAMES) + ""));

            // effectively grab and process a single frame
            imgProcess.getOriginalFrame().copyTo(originalFrame);
            drawRect(originalFrame);

            try {
                CSVWriter.writeLine(writer, evaluation());
            } catch (IOException e) {
                System.out.print("Stream closed");
            }
        }

    }

    @FXML
    protected void save(ActionEvent event) {
        try {
            for (int i = 0; i < list.size(); i++) {
                CSVWriter.writeLine(writer, list.get(i).toCSVFormat());
            }
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Information Dialog");
            alert.setHeaderText("Look, an Information Dialog");
            alert.setContentText("Save done!");

            alert.showAndWait();
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (NullPointerException ex) {
        }
    }


    private List evaluation() {
        if (person_there.isSelected()) {
            return Arrays.asList(
                    frame_number.getText(),
                    bb_values.getText(),
                    person_there.isSelected() + "",
                    posture.getSelectedToggle().getUserData().toString(),
                    isMoving.getSelectedToggle().getUserData().toString(),
                    status.getSelectedToggle().getUserData().toString()
            );
        } else {
            return Arrays.asList(frame_number.getText());
        }
    }


    private void stopAcquisition() {
        try {
            for (int i = 0; i < list.size(); i++) {
                CSVWriter.writeLine(writer, list.get(i).toCSVFormat());
            }
            writer.flush();
            writer.close();
        } catch (IOException e) {
            System.out.print("Stream close!");
        } catch (NullPointerException ex) {
            System.out.print("Stream close!");
        }
        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                // log any exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }
        }

        if (this.capture.isOpened()) {
            // release the camera
            this.capture.release();
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

    public void setClosed() {
        this.stopAcquisition();
    }

    public void startDrawRect(MouseEvent mouseEvent) {
        if (mouseEvent.getX() < originalFrame.width() && mouseEvent.getY() < originalFrame.height() &&
                mouseEvent.getX() >= 0 && mouseEvent.getY() >= 0) {
            startX = mouseEvent.getX();
            startY = mouseEvent.getY();
        }
    }

    public void drawRect(MouseEvent mouseEvent) {
        if (mouseEvent.getX() < originalFrame.width() && mouseEvent.getY() < originalFrame.height() &&
                mouseEvent.getX() >= 0 && mouseEvent.getY() >= 0) {
            endX = mouseEvent.getX();
            endY = mouseEvent.getY();
            drawRect(originalFrame);
        }
    }

    public void stopDrawRect(MouseEvent mouseEvent) {
        if (mouseEvent.getX() < originalFrame.width() && mouseEvent.getY() < originalFrame.height() &&
                mouseEvent.getX() >= 0 && mouseEvent.getY() >= 0) {
            endX = mouseEvent.getX();
            endY = mouseEvent.getY();
            drawRect(originalFrame);
        }
    }


    private void ini() {
        //Set extension filter
        FileChooser.ExtensionFilter extFilter = new FileChooser.ExtensionFilter("Video files (*.avi, *.mp4)", "*.avi", "*.mp4");
        fileChooser.getExtensionFilters().add(extFilter);
        //Show open file dialog
        file = fileChooser.showOpenDialog(null);

        person_there.setSelected(false);
        person_there.selectedProperty().addListener((observableValue, old_value, new_value) ->
        {
            if (new_value == false) {
                resetAllValue();
            }
            if (new_value == true) {
                initAllValue();
            }
        });

        standing.setToggleGroup(posture);
        standing.setUserData("standing");
        bending.setToggleGroup(posture);
        bending.setUserData("bending");
        sitting.setToggleGroup(posture);
        sitting.setUserData("sitting");
        laying.setToggleGroup(posture);
        laying.setUserData("laying");


        moving.setToggleGroup(isMoving);
        moving.setUserData("moving");
        not_moving.setToggleGroup(isMoving);
        not_moving.setUserData("not_moving");

        ok.setToggleGroup(status);
        ok.setUserData("ok");
        not_ok.setToggleGroup(status);
        not_ok.setUserData("not_ok");
    }

    private void resetAllValue() {
        startX = 0;
        startY = 0;
        endX = 0;
        endY = 0;
        setTextOnBBLabel();
        posture.selectToggle(null);
        isMoving.selectToggle(null);
        status.selectToggle(null);
        drawRect(originalFrame);

        standing.setDisable(true);
        bending.setDisable(true);
        sitting.setDisable(true);
        laying.setDisable(true);
        moving.setDisable(true);
        not_moving.setDisable(true);
        ok.setDisable(true);
        not_ok.setDisable(true);
    }

    private void initAllValue() {
        standing.setDisable(false);
        bending.setDisable(false);
        sitting.setDisable(false);
        laying.setDisable(false);
        moving.setDisable(false);
        not_moving.setDisable(false);
        ok.setDisable(false);
        not_ok.setDisable(false);

        standing.fire();
        moving.fire();
        ok.fire();
    }

    private void setTextOnBBLabel() {
        Platform.runLater(() -> bb_values.setText(startX + "," + startY + "," + endX + "," + endY));
    }

    private void createCVSFile() {
        String csvFile = fileName + ".csv";
        try {
            writer = new FileWriter(csvFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void iniTimerBar() {
        if (this.capture.isOpened()) {
            list = new ArrayList<>();
            for (int i = 0; i < this.capture.get(Videoio.CAP_PROP_FRAME_COUNT); i++) {
                EvaluationValue evaluationValue = new EvaluationValue();
                evaluationValue.setFrame_number((i + 1) + "");
                list.add(evaluationValue);
            }

            timerbar.setMin(0);
            timerbar.setMax(this.capture.get(Videoio.CAP_PROP_FRAME_COUNT));
            timerbar.setValue(0);
            timerbar.valueProperty().addListener((observable, oldValue, newValue) -> {
                if (newValue.intValue() <= this.capture.get(Videoio.CAP_PROP_FRAME_COUNT)) {
                    Platform.runLater(() -> frame_number.setText(capture.get(Videoio.CAP_PROP_POS_FRAMES) + ""));
                    capture.set(Videoio.CAP_PROP_POS_FRAMES, newValue.doubleValue());
                    imgProcess.getOriginalFrame().copyTo(originalFrame);
                    Platform.runLater(() -> updateImageView(imageView, Utils.mat2Image(originalFrame)));

                    int startindex = oldValue.intValue();
                    int length = newValue.intValue() - oldValue.intValue();
                    int sign = 1;
                    if (length < 0) {
                        sign = -1;
                        length = Math.abs(length);
                    }

                    for (int i = 0; i < length; i++) {
                        if (person_there.isSelected() && startindex >= 0 && startindex < list.size()) {
                            list.get(startindex).setPerson_there(person_there.isSelected() + "");
                            list.get(startindex).setPosture(posture.getSelectedToggle().getUserData().toString());
                            list.get(startindex).setIsMoving(isMoving.getSelectedToggle().getUserData().toString());
                            list.get(startindex).setStatus(status.getSelectedToggle().getUserData().toString());
                        }

                        startindex = startindex + sign;
                    }
                }
            });
        }
    }

}
