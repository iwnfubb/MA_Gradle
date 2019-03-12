package guicontroler;

import imageprocess.ImageProcessObjectTracking;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;
import utils.CSVWriter;
import utils.EvaluationValue;
import utils.Parameters;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

public class NewGUIControllerObjectTracking {
    private static final Logger LOG = Logger.getLogger(NewGUIControllerObjectTracking.class.getName());
    // the id of the camera to be used
    private static int cameraId = 0;
    private static AtomicInteger frameCounter = new AtomicInteger();
    FileWriter fileWriter;
    FileWriter fileWriterTimer;
    ArrayList<EvaluationValue> list;
    ArrayList<String> listTimer;
    Iterator<File> iterator = getAllFilesInFolder().iterator();
    Iterator<Double> iteratorParameters = Parameters.roc_parameters.iterator();
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
    ImageProcessObjectTracking imgProcess = new ImageProcessObjectTracking(this.capture);
    // a flag to change the button behavior
    private boolean cameraActive = false;
    private boolean trigger = false;
    private boolean liveVideo = true;
    private boolean needToSaveRecord = false;
    private Timestamp timestamp = new Timestamp(System.currentTimeMillis());
    private String fileName = "v_fallen2.mp4";
    private String inputPath = Utils.PATH_TO_VIDEOS_INPUT_FOLDER + fileName;
    private String outputPath;
    private int outputWidth = 1280 * 3;
    private int outputHeight = 720 * 3;
    private VideoWriter videoWriter;
    private boolean started = false;

    /**
     * The action triggered by pushing the button on the GUI
     */
    @FXML
    protected void startCamera() {
        iterateThresholds();
        String fileNameWithoutExt = iteratePlaybackVideos();
        //If there is no videos more then return, we are finished
        if (fileNameWithoutExt == null) {
            return;
        }
        checkIfShadowRemoverActivated(fileNameWithoutExt);
        initOutputMP4Data();
        if (!this.cameraActive) {
            // start the video capture
            startCapture(liveVideo);
            setTimeBarIfPlaybackVideo();
            // is the video stream available?
            if (this.capture.isOpened()) {
                this.cameraActive = true;
                Runnable frameGrabber = () -> {
                    iniPlaybackVideos(liveVideo);
                    // effectively grab and process a single frame
                    Mat originalFrame = imgProcess.getOriginalFrame();
                    resizeOriginalFrame(originalFrame, 600, 400);
                    //If a playback video end then restart the capture and get the next video in the folder
                    restartCaptureForPlaybackVideos(originalFrame);
                    Mat output = processImage(originalFrame);
                    saveToMp4AndCsvDataIfNeeded(output, needToSaveRecord);
                };
                // Set time to 33 milliseconds to get 30 frames per second
                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
                // update the button content
                this.button.setText("Stop Camera");
            } else {
                // log the error
                LOG.info("Impossible to open the camera connection...");
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

    private void initOutputMP4Data() {
        if (needToSaveRecord) {
            videoWriter = new VideoWriter(outputPath, VideoWriter.fourcc('D', 'I', 'V', 'X'), 30, new Size(outputWidth, outputHeight), true);
        }
    }

    private Mat processImage(Mat originalFrame) {
        Mat[] detection = imgProcess.personDetector(originalFrame);
        Mat firstRow = Utils.hstack(Utils.hstack(originalFrame, originalFrame), originalFrame);
        Mat secondRow = Utils.hstack(Utils.hstack(originalFrame, originalFrame), originalFrame);
        firstRow = Utils.vstack(firstRow, secondRow);
        Mat thirdRow = Utils.hstack(Utils.hstack(originalFrame, originalFrame), originalFrame);
        firstRow = Utils.vstack(firstRow, thirdRow);
        Image image = Utils.mat2Image(firstRow);
        updateImageView(imageView, image);
        return Utils.rescaleImageToDisplay(firstRow, outputWidth, outputHeight);
    }

    private void saveToMp4AndCsvDataIfNeeded(Mat output, final boolean needToSaveRecord) {
        if (needToSaveRecord) {
            videoWriter.write(output);
            if (imgProcess.getProcessTime() != 0) {
                listTimer.add(String.valueOf(imgProcess.getProcessTime()));
            }
        }
    }

    private void restartCaptureForPlaybackVideos(Mat originalFrame) {
        if (!liveVideo && originalFrame.empty()) {
            this.cameraActive = false;
            save();
            this.capture.release();
            if (needToSaveRecord) {
                videoWriter.release();
            }
            imgProcess.iniBackground(false);
            startCamera();
        }
    }

    private void resizeOriginalFrame(Mat originalFrame, final int width, final int height) {
        if (!originalFrame.empty() && originalFrame.width() > 0 && originalFrame.height() > 0) {
            Imgproc.resize(originalFrame, originalFrame, new Size(width, height));
        }
    }

    private void iniPlaybackVideos(final boolean isLive) {
        if (!isLive) {
            if (trigger) {
                capture.set(Videoio.CAP_PROP_POS_FRAMES, frameCounter.get());
                trigger = false;
            } else {
                timerbar.setValue(frameCounter.incrementAndGet());
                trigger = false;
            }
            //set frame number for play back video, dont need on live video
            imgProcess.setFrameNumber((int) capture.get(Videoio.CAP_PROP_POS_FRAMES));
        }
    }

    private void startCapture(final boolean isLive) {
        if (isLive) {
            this.capture.open(cameraId);
        } else {
            this.capture.open(inputPath);
        }
    }

    private void checkIfShadowRemoverActivated(String fileNameWithoutExt) {
        if (Utils.activeShadowRemover) {
            outputPath = Utils.PATH_TO_VIDEOS_OUTPUT_FOLDER + timestamp.getTime() + "vo_noshadow" + fileNameWithoutExt + "_" + Parameters.badValue + "_" + ".mp4";
        } else {
            outputPath = Utils.PATH_TO_VIDEOS_OUTPUT_FOLDER + timestamp.getTime() + "vo_" + fileNameWithoutExt + "_" + Parameters.badValue + "_" + ".mp4";
        }
    }

    private String iteratePlaybackVideos() {
        File f;
        if (iteratorParameters.hasNext() || iterator.hasNext()) {
            if (iterator.hasNext()) {
                f = iterator.next();
            } else {
                Parameters.badValue = iteratorParameters.next();
                LOG.log(Level.INFO, " ###############################################################\n" +
                        "########## Test with Parameter badValue = \" {} \"##########\n"
                        + "###############################################################", Parameters.badValue);
                iterator = getAllFilesInFolder().iterator();
                f = iterator.next();
            }
        } else {
            try {
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                LOG.info("Error by stop videos playback");
                Thread.currentThread().interrupt();
            }
            return null;
        }

        inputPath = f.getPath();
        fileName = f.getName();
        String fileNameWithoutExt = fileName;
        if (fileNameWithoutExt.contains(".")) {
            fileNameWithoutExt = fileNameWithoutExt.substring(0, fileNameWithoutExt.lastIndexOf('.'));
        }
        return fileNameWithoutExt;
    }

    private void iterateThresholds() {
        if (!started && iteratorParameters.hasNext()) {
            Parameters.badValue = iteratorParameters.next();
            LOG.log(Level.INFO, " ###############################################################\n" +
                    "########## Test with Parameter badValue = \" {} \"##########\n"
                    + "###############################################################", Parameters.badValue);
            started = true;
        }
    }


    @FXML
    protected void startTracking(ActionEvent event) {
        if (!imgProcess.isPersonTracking()) {
            imgProcess.startPersonTracking();
            button2.setText("Stop tracking");
        } else {
            imgProcess.stopPersonTracking();
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
                LOG.info("Exception in stopping the frame capture, trying to release the camera now... " + e);
                Thread.currentThread().interrupt();
            }
        }
        if (this.capture.isOpened()) {
            // release the camera
            this.capture.release();
            if (!liveVideo) {
                videoWriter.release();
            }
        }
    }

    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

    public void setClosed() {
        this.stopAcquisition();
    }

    private void setTimeBarIfPlaybackVideo() {
        if (!liveVideo) {
            timerbar.setMin(0);
            timerbar.setMax(this.capture.get(Videoio.CAP_PROP_FRAME_COUNT));
            timerbar.valueProperty().addListener((observable, oldValue, newValue) -> {
                if (newValue.intValue() < this.capture.get(Videoio.CAP_PROP_FRAME_COUNT)) {
                    frameCounter.set(newValue.intValue());
                    trigger = true;
                }
            });
        }
    }

    private void save() {
        try {
            for (int i = 0; i < list.size(); i++) {
                CSVWriter.writeLine(fileWriter, list.get(i).toCSVFormat());
            }
            for (String str : listTimer) {
                ArrayList<String> l = new ArrayList<>();
                l.add(str);
                CSVWriter.writeLine(fileWriterTimer, l);
            }
            fileWriter.flush();
            fileWriter.close();
            fileWriterTimer.flush();
            fileWriterTimer.close();
        } catch (IOException e) {
            LOG.info("Error by saving file " + e);
        }
    }


    private List<File> getAllFilesInFolder() {
        File folder = new File(Utils.PATH_TO_VIDEOS_INPUT_FOLDER);
        File[] listOfFiles = folder.listFiles((dir, name) -> name.endsWith(".mp4"));

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                LOG.info("File " + listOfFiles[i].getName());
            } else if (listOfFiles[i].isDirectory()) {
                LOG.info("Directory " + listOfFiles[i].getName());
            }
        }
        return Arrays.asList(listOfFiles);
    }
}
