package guicontroler;

import imageprocess.ImageProcess_ObjectTracking;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;
import utils.Utils;

import javax.rmi.CORBA.Util;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class NewGUIController_ObjectTracking {
    @FXML
    private CheckBox grabcutActive;
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
    private String fileName = "v_fallen.mp4";
    private String outputName = "vo_" + fileName;
    private int output_width = 1280;
    private int output_height = 720;
    private VideoWriter writer = new VideoWriter(outputName, VideoWriter.fourcc('D', 'I', 'V', 'X'), 1, new Size(output_width, output_height), true);


    /**
     * The action triggered by pushing the button on the GUI
     *
     * @param event the push button event
     */
    @FXML
    protected void startCamera(ActionEvent event) {
        frameCounter = 0;
        if (!this.cameraActive) {

            // start the video capture
            if (liveVideo) {
                this.capture.open(cameraId);
            } else {
                this.capture.open(fileName);
            }
            ini();
            // is the video stream available?
            if (this.capture.isOpened()) {
                this.cameraActive = true;

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

                    // effectively grab and process a single frame
                    Mat originalFrame = imgProcess.getOriginalFrame();
                    Mat gaussianBlurFrame = imgProcess.getGaussianBlur(originalFrame);

                    Mat firstRow = Utils.hstack(originalFrame, gaussianBlurFrame);

                    if (!gaussianBlurFrame.empty() && grabcutActive.isSelected()) {
                        Mat[] grabcutFrameAndMatches = imgProcess.personDetector(originalFrame);
                        Mat secondRow = Utils.hstack(Utils.hstack(grabcutFrameAndMatches[0], grabcutFrameAndMatches[1]), grabcutFrameAndMatches[2]);
                        firstRow = Utils.vstack(firstRow, secondRow);

                        if (grabcutFrameAndMatches.length == 6) {
                            Mat thirdRow = Utils.hstack(Utils.hstack(grabcutFrameAndMatches[3], grabcutFrameAndMatches[4]), grabcutFrameAndMatches[5]);
                            firstRow = Utils.vstack(firstRow, thirdRow);
                        }
                    }
                    Image image = Utils.mat2Image(firstRow);
                    updateImageView(imageView, image);
                    Mat output = Utils.rescaleImageToDisplay(firstRow, output_width, output_height);
                    writer.write(output);
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
            writer.release();
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

}
