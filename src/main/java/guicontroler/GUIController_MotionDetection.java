package guicontroler;

import imageprocess.ImageProcess_MotionDetection;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import utils.Utils;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class GUIController_MotionDetection {
    @FXML
    private ImageView currentFrameView;
    @FXML
    private ImageView gmmFrameView;
    @FXML
    private ImageView gaussianBlurView;
    @FXML
    private ImageView knnView;
    @FXML
    private ImageView kdeView;
    @FXML
    private ImageView vibeView;
    @FXML
    private CheckBox gmmActive;
    @FXML
    private CheckBox knnActive;
    @FXML
    private CheckBox kdeActive;
    @FXML
    private CheckBox vibeActive;
    @FXML
    private Slider gmmHistory;
    @FXML
    private Slider knnHistory;
    @FXML
    private Slider gaussianBlur;
    @FXML
    private Slider kdeThreshold;
    @FXML
    private Slider vibeThreshold;
    @FXML
    private Button button;

    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that realizes the video capture
    private VideoCapture capture = new VideoCapture();
    ImageProcess_MotionDetection imgProcess = new ImageProcess_MotionDetection(this.capture);
    // a flag to change the button behavior
    private boolean cameraActive = false;
    // the id of the camera to be used
    private static int cameraId = 0;


    /**
     * The action triggered by pushing the button on the GUI
     *
     * @param event the push button event
     */
    @FXML
    protected void startCamera(ActionEvent event) {
        ini();
        if (!this.cameraActive) {
            // start the video capture
            //this.capture.open(cameraId);
            this.capture.open(Utils.PATH_TO_VIDEOS_INPUT_FOLDER + "v_dead.mp4");

            // is the video stream available?
            if (this.capture.isOpened()) {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = () -> {
                    // effectively grab and process a single frame
                    Mat originalFrame = imgProcess.getOriginalFrame();
                    Mat gaussianBlurFrame = imgProcess.getGaussianBlur();

                    Image imageToShow = Utils.mat2Image(originalFrame);
                    updateImageView(currentFrameView, imageToShow);

                    if (!gaussianBlurFrame.empty()) {
                        Image mmgImageToShow = Utils.mat2Image(gaussianBlurFrame);
                        updateImageView(gaussianBlurView, mmgImageToShow);
                    }

                    if (gmmActive.isSelected()) {
                        Mat gmmFrame = imgProcess.getGaussianMixtureModel();
                        if (!gmmFrame.empty()) {
                            Image mmgImageToShow = Utils.mat2Image(gmmFrame);
                            updateImageView(gmmFrameView, mmgImageToShow);
                        }
                    }

                    if (knnActive.isSelected()) {
                        Mat knnFrame = imgProcess.getKNNModel();
                        if (!knnFrame.empty()) {
                            Image mmgImageToShow = Utils.mat2Image(knnFrame);
                            updateImageView(knnView, mmgImageToShow);
                        }
                    }

                    if (kdeActive.isSelected()) {
                        Mat kdeFrame = imgProcess.getKDEModel();
                        if (!kdeFrame.empty()) {
                            Image mmgImageToShow = Utils.mat2Image(kdeFrame);
                            updateImageView(kdeView, mmgImageToShow);
                        }
                    }

                    if (vibeActive.isSelected()) {
                        Mat vibeFrame = imgProcess.getVibeModel();
                        if (!vibeFrame.empty()) {
                            Image mmgImageToShow = Utils.mat2Image(vibeFrame);
                            updateImageView(vibeView, mmgImageToShow);
                        }
                    }
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
    protected void setStaticBackground(ActionEvent event) {
        System.err.println("Not supported");
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
        }
    }

    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

    public void setClosed() {
        this.stopAcquisition();
    }

    private void ini() {
        gmmHistory.setMin(0);
        gmmHistory.setMax(500);
        gmmHistory.setValue(100);
        gmmHistory.valueProperty().addListener((observable, oldValue, newValue) -> {
            imgProcess.setHistoryGMM(newValue.intValue());
        });

        knnHistory.setMin(0);
        knnHistory.setMax(500);
        knnHistory.setValue(100);
        knnHistory.valueProperty().addListener((observable, oldValue, newValue) -> {
            imgProcess.setHistoryGMM(newValue.intValue());
        });

        gaussianBlur.setMin(1);
        gaussianBlur.setMax(45);
        gaussianBlur.valueProperty().addListener((observable, oldValue, newValue) -> {
            imgProcess.setGaussianFilterSize(newValue.intValue());
        });

        kdeThreshold.setMin(-1.0);
        kdeThreshold.setMax(1.0);
        kdeThreshold.valueProperty().addListener((observable, oldValue, newValue) -> {
            imgProcess.setKDEThreshole(newValue.doubleValue());
        });

        vibeThreshold.setMin(-50.0);
        vibeThreshold.setMax(50.0);
        vibeThreshold.valueProperty().addListener((observable, oldValue, newValue) -> {
            imgProcess.setKDEThreshole(newValue.doubleValue());
        });
    }

}
