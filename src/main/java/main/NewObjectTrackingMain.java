package main;

import guicontroler.NewGUIControllerObjectTracking;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import org.opencv.core.Core;

public class NewObjectTrackingMain extends Application {
    @Override
    public void start(Stage primaryStage) throws Exception {
        try {
            FXMLLoader root = new FXMLLoader(getClass().getResource("../../resources/newobjecttracking.fxml"));
            BorderPane rootElement = root.load();

            primaryStage.setTitle("Object Tracking");
            primaryStage.setScene(new Scene(rootElement, 1200, 800));
            primaryStage.setResizable(false);
            primaryStage.show();

            NewGUIControllerObjectTracking controller = root.getController();
            primaryStage.setOnCloseRequest((we -> controller.setClosed()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        launch(args);
    }
}
