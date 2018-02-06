package utils;

import java.util.Arrays;
import java.util.List;

public class EvaluationValue {
    String frame_number;
    String bb_values;
    String person_there;
    String posture;
    String isMoving;
    String status;

    public EvaluationValue() {
        frame_number = "0";
        bb_values = "0,0,0,0";
        person_there = "false";
        posture = "";
        isMoving = "";
        status = "";
    }

    public String getFrame_number() {
        return frame_number;
    }

    public double getFrame_number_in_double() {
        return Double.parseDouble(frame_number);
    }

    public void setFrame_number(String frame_number) {
        this.frame_number = frame_number;
    }

    public String getBb_values() {
        return bb_values;
    }

    public void setBb_values(String bb_values) {
        this.bb_values = bb_values;
    }

    public String getPerson_there() {
        return person_there;
    }

    public boolean getPerson_there_in_boolean() {
        return Boolean.parseBoolean(person_there);
    }

    public void setPerson_there(String person_there) {
        this.person_there = person_there;
    }

    public String getPosture() {
        return posture;
    }

    public double getPosture_in_double() {
        if (posture.equals("standing")) {
            return 0;
        }
        if (posture.equals("laying")) {
            return 1;
        }
        if (posture.equals("bending")) {
            return 2;
        }
        if (posture.equals("sitting")) {
            return 3;
        }
        return -1;
    }

    public void setPosture(String posture) {
        this.posture = posture;
    }

    public String getIsMoving() {
        return isMoving;
    }

    public boolean getIsMoving_in_boolean() {
        if (isMoving.equals("moving")) {
            return true;
        }
        return false;
    }

    public void setIsMoving(String isMoving) {
        this.isMoving = isMoving;
    }

    public String getStatus() {
        return status;
    }

    public boolean getStatusIsOk_in_boolean() {
        if (status.equals("ok")) {
            return true;
        }
        return false;
    }


    public void setStatus(String status) {
        this.status = status;
    }

    public List toCSVFormat() {
        if (person_there.compareTo("true") == 0) {
            return Arrays.asList(
                    frame_number,
                    bb_values,
                    person_there + "",
                    posture,
                    isMoving,
                    status
            );
        } else {
            return Arrays.asList(frame_number);
        }
    }


}
