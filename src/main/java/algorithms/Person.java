package algorithms;

import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Iterator;

public class Person {
    private static int counter = 0;
    Rect rect;
    int movementMaximum;  //amount to move to still be the same person
    int movementMinimum;   //minimum amount to move to not trigger alarm
    int movementTime;     //number of frames after the alarm is triggered
    int lastmoveTime = 0;
    int lastseenTime = 0;
    int id = counter++;

    boolean alert;
    boolean remove;
    String posture;
    double bad_prediction;
    double good_prediction;

    public Person(Rect rect, int movementMaximum, int movementMinimum, int movementTime) {
        this.rect = rect;
        this.movementMaximum = movementMaximum;
        this.movementMinimum = movementMinimum;
        this.movementTime = movementTime;
    }

    public boolean samePerson(Rect rect) {
        if (rect.x + movementMaximum > this.rect.x && rect.x - movementMaximum < this.rect.x) {
            if (rect.y + movementMaximum > this.rect.y && rect.y - movementMaximum < this.rect.y) {
                return true;
            }
        }
        return false;
    }

    public void editPerson(Rect rect) {
        //if (MovingDetector.isObjectMoving(rect, this.rect))
        if (Math.abs(rect.x - this.rect.x) > movementMinimum ||
                Math.abs(rect.y - this.rect.y) > movementMinimum ||
                Math.abs(rect.width - this.rect.width) > movementMinimum ||
                Math.abs(rect.height - this.rect.height) > movementMinimum) {
            lastmoveTime = 0;
        }
        this.rect = rect;
        this.lastseenTime = 0;
    }


    public int getID() {
        return id;
    }

    public void tick() {
        lastmoveTime += 1;
        lastseenTime += 1;
        if (lastmoveTime > movementTime && posture == "laying" && bad_prediction > 0.8) {
            alert = true;
        }
        if (lastseenTime > 4) {
            remove = true;
        }
    }

    public boolean getAlert() {
        return alert;
    }

    public boolean getRemove() {
        return remove;
    }

    public static class Persons {
        ArrayList<Person> persons = new ArrayList<>();
        int movementMaximum;
        int movementMinimum;
        int movementTime;

        public Persons(int movementMaximum, int movementMinimum, int movementTime) {
            this.movementMaximum = movementMaximum;
            this.movementMinimum = movementMinimum;
            this.movementTime = movementTime;
        }

        public Person addPerson(Rect rect) {
            Person p = familiarPerson(rect);
            if (p != null) {
                p.editPerson(rect);
                return p;
            } else {
                Person new_person = new Person(rect, movementMaximum, movementMinimum, movementTime);
                persons.add(new_person);
                return new_person;
            }
        }

        public Person familiarPerson(Rect rect) {
            for (Person p : persons) {
                if (p.samePerson(rect)) {
                    return p;
                }
            }
            return null;
        }

        public void tick() {
            Iterator<Person> iterator = persons.iterator();
            while (iterator.hasNext()) {
                Person p = iterator.next();
                p.tick();
                if (p.getRemove()) {
                    iterator.remove();
                }
            }
        }
    }


}
