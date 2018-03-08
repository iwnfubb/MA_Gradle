package guicontroler;

import imageprocess.ImageProcess_ObjectTracking;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;
import utils.CSVWriter;
import utils.EvaluationValue;
import utils.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class NewGUIController_ObjectTracking {
    @FXML
    private Slider timerbar;
    @FXML
    private Button button;
    @FXML
    private Button button2;
    @FXML
    private ImageView imageView;

    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that realizes the video capture
    private VideoCapture capture = new VideoCapture();
    ImageProcess_ObjectTracking imgProcess = new ImageProcess_ObjectTracking(this.capture);
    // a flag to change the button behavior
    private boolean cameraActive = false;
    // the id of the camera to be used
    private static int cameraId = 0;

    private Mat previousFrameFlow = new Mat();

    private boolean trigger = false;
    private boolean liveVideo = false;
    public static int frameCounter = 0;
    private Timestamp timestamp = new Timestamp(System.currentTimeMillis());

    private String fileName = "v_fallen2.mp4";
    private String inputPath = Utils.PATH_TO_VIDEOS_INPUT_FOLDER + fileName;
    private long timeStamp = timestamp.getTime();
    private String outputPath;
    private int output_width = 1280 * 3;
    private int output_height = 720 * 3;
    private VideoWriter videoWriter;
    FileWriter fileWriter;
    ArrayList<EvaluationValue> list;
    Iterator<File> iterator = getAllFilesInFolder().iterator();


    /**
     * The action triggered by pushing the button on the GUI
     */
    @FXML
    protected void startCamera() {
        File f = null;
        if (iterator.hasNext()) {
            f = iterator.next();
        } else {
            try {
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return;
        }
        //files = getAllFilesInFolder();

        inputPath = f.getPath();
        fileName = f.getName();
        String fileNameWithoutExt = fileName;
        if (fileNameWithoutExt.indexOf(".") > 0) {
            fileNameWithoutExt = fileNameWithoutExt.substring(0, fileNameWithoutExt.lastIndexOf("."));
        }
        if (Utils.activeShadowRemover) {
            outputPath = Utils.PATH_TO_VIDEOS_OUTPUT_FOLDER + timeStamp + "vo_noshadow" + fileNameWithoutExt + ".mp4";
        } else {
            outputPath = Utils.PATH_TO_VIDEOS_OUTPUT_FOLDER + timeStamp + "vo_" + fileNameWithoutExt + ".mp4";
        }

        videoWriter = new VideoWriter(outputPath, VideoWriter.fourcc('D', 'I', 'V', 'X'), 30, new Size(output_width, output_height), true);

        frameCounter = 0;
        if (!this.cameraActive) {

            // start the video capture
            if (liveVideo) {
                this.capture.open(cameraId);
            } else {
                this.capture.open(inputPath);
            }
            ini();
            // is the video stream available?
            if (this.capture.isOpened()) {
                this.cameraActive = true;
                if (!liveVideo) {
                    createCVSFile();
                }

                Runnable frameGrabber = () -> {
                    if (!liveVideo) {
                        if (trigger) {
                            capture.set(Videoio.CAP_PROP_POS_FRAMES, frameCounter);
                            trigger = false;
                        } else {
                            timerbar.setValue(++frameCounter);
                            trigger = false;
                        }
                    }
                    imgProcess.personDetectorAndTracking.frame_number = (int) capture.get(Videoio.CAP_PROP_POS_FRAMES);
                    // effectively grab and process a single frame
                    Mat originalFrame = imgProcess.getOriginalFrame();
                    if (originalFrame.empty()) {
                        this.cameraActive = false;
                        save();
                        this.capture.release();
                        this.videoWriter.release();
                        imgProcess.personDetectorAndTracking.diffMotionDetector.isBackgroundSet = false;
                        startCamera();
                    }
                    Mat firstRow;

                    Mat[] detection = imgProcess.personDetector(originalFrame);
                    firstRow = Utils.hstack(Utils.hstack(detection[0], detection[1]), detection[2]);
                    if (detection.length >= 6) {
                        Mat secondRow = Utils.hstack(Utils.hstack(detection[3], detection[4]), detection[5]);
                        firstRow = Utils.vstack(firstRow, secondRow);
                    }
                    if (detection.length == 9) {
                        Mat thirdRow = Utils.hstack(Utils.hstack(detection[6], detection[7]), detection[8]);
                        firstRow = Utils.vstack(firstRow, thirdRow);
                    }
                    Image image = Utils.mat2Image(firstRow);
                    updateImageView(imageView, image);
                    Mat output = Utils.rescaleImageToDisplay(firstRow, output_width, output_height);
                    videoWriter.write(output);
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                // update the button content
                this.button.setText("Stop Camera");
            } else {
                // log the error
                System.err.println("Impossible to open the camera connection...");
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.button.setText("Start Camera");

            // stop the timer
            this.stopAcquisition();
        }
    }


    @FXML
    protected void startTracking(ActionEvent event) {
        if (!imgProcess.personDetectorAndTracking.isTracking()) {
            imgProcess.personDetectorAndTracking.startTracking();
            button2.setText("Stop tracking");
        } else {
            imgProcess.personDetectorAndTracking.stopTracking();
            button2.setText("Start tracking");
        }

    }

    private void stopAcquisition() {
        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                if (!liveVideo) {
                    save();
                }
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
            videoWriter.release();

        }
    }

    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

    public void setClosed() {
        this.stopAcquisition();
    }

    private void ini() {
        if (!liveVideo) {
            timerbar.setMin(0);
            timerbar.setMax(this.capture.get(Videoio.CAP_PROP_FRAME_COUNT));
            timerbar.valueProperty().addListener((observable, oldValue, newValue) -> {
                if (newValue.intValue() < this.capture.get(Videoio.CAP_PROP_FRAME_COUNT)) {
                    this.frameCounter = newValue.intValue();
                    this.trigger = true;
                }
            });
        }
    }


    private void createCVSFile() {
        list = new ArrayList<>();
        String fileNameWithoutExt = fileName;
        if (fileNameWithoutExt.indexOf(".") > 0) {
            fileNameWithoutExt = fileNameWithoutExt.substring(0, fileNameWithoutExt.lastIndexOf("."));
        }
        String csvFile;
        if (Utils.activeShadowRemover) {
            csvFile = Utils.PATH_TO_VIDEOS_OUTPUT_FOLDER + timeStamp + fileNameWithoutExt + "_noshadow.csv";
        } else {
            csvFile = Utils.PATH_TO_VIDEOS_OUTPUT_FOLDER + timeStamp + fileNameWithoutExt + ".csv";
        }
        try {
            fileWriter = new FileWriter(csvFile);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < this.capture.get(Videoio.CAP_PROP_FRAME_COUNT); i++) {
            EvaluationValue evaluationValue = new EvaluationValue();
            evaluationValue.setFrame_number((i + 1) + "");
            list.add(evaluationValue);
        }
        imgProcess.personDetectorAndTracking.list = this.list;
    }

    private void save() {
        try {
            for (int i = 0; i < list.size(); i++) {
                CSVWriter.writeLine(fileWriter, list.get(i).toCSVFormat());
            }
            /*Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Information Dialog");
            alert.setHeaderText("Look, an Information Dialog");
            alert.setContentText("Save done!");

            alert.showAndWait();*/
            fileWriter.flush();
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (NullPointerException ex) {
        }
    }


    private List<File> getAllFilesInFolder() {
        File folder = new File(Utils.PATH_TO_VIDEOS_INPUT_FOLDER);
        File[] listOfFiles = folder.listFiles((dir, name) -> name.endsWith(".mp4"));

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                System.out.println("File " + listOfFiles[i].getName());
            } else if (listOfFiles[i].isDirectory()) {
                System.out.println("Directory " + listOfFiles[i].getName());
            }
        }
        List<File> files = Arrays.asList(listOfFiles);
        return files;
    }
}
