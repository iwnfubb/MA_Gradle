package main;

import guicontroler.GUIController_VideoAnnotation;
import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import org.opencv.core.Core;

public class VideoAnnotation extends Application {
    @Override
    public void start(Stage primaryStage) throws Exception {
        try {
            FXMLLoader root = new FXMLLoader(getClass().getResource("../../resources/videoannotation.fxml"));
            BorderPane rootElement = root.load();

            primaryStage.setTitle("Video Annotation");
            primaryStage.setScene(new Scene(rootElement, 1600, 912));
            primaryStage.setResizable(false);
            primaryStage.show();

            GUIController_VideoAnnotation controller = root.getController();
            primaryStage.setOnCloseRequest((we -> controller.setClosed()));

            primaryStage.addEventFilter(MouseEvent.MOUSE_PRESSED, (me -> controller.startDrawRect(me)));
            primaryStage.addEventFilter(MouseEvent.MOUSE_DRAGGED, (me -> controller.drawRect(me)));
            primaryStage.addEventFilter(MouseEvent.MOUSE_RELEASED, (me -> controller.stopDrawRect(me)));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        launch(args);
    }
}
