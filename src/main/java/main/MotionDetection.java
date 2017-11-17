package main;

import guicontroler.GUIController_MotionDetection;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import org.opencv.core.Core;

public class MotionDetection extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        try {
            FXMLLoader root = new FXMLLoader(getClass().getResource("../../resources/motiondetection.fxml"));
            BorderPane rootElement = root.load();

            primaryStage.setTitle("Motion Detection");
            primaryStage.setScene(new Scene(rootElement, 800, 800));
            primaryStage.setResizable(false);
            primaryStage.show();

            GUIController_MotionDetection GUIControllerMotionDetection = root.getController();
            primaryStage.setOnCloseRequest((we -> GUIControllerMotionDetection.setClosed()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        launch(args);
    }
}
