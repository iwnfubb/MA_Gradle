package algorithms;

import org.opencv.core.Rect;
import utils.Parameters;

import java.util.ArrayList;
import java.util.Iterator;

public class Person {
    private static int counter = 0;
    Rect rect;
    int movementMaximum;  //amount to move to still be the same person
    int movementMinimum;   //minimum amount to move to not trigger alarm
    int badLimit;     //number of frames after the alarm is triggered
    int lastmoveTime = 0;
    int badCounter = 0;
    int lastseenTime = 0;
    int id = counter++;
    int sameBBDetected = 0;

    public boolean forcedDelete;
    boolean alert;
    boolean remove;
    String posture;
    double bad_prediction;
    double good_prediction;

    public Person(Rect rect, int movementMaximum, int movementMinimum, int badLimit) {
        this.rect = rect;
        this.movementMaximum = movementMaximum;
        this.movementMinimum = movementMinimum;
        this.badLimit = badLimit;
    }

    public Person clone() {
        Person person = new Person(rect, movementMaximum, movementMinimum, badLimit);
        person.lastseenTime = lastseenTime;
        person.lastmoveTime = lastmoveTime;
        person.id = id;
        person.sameBBDetected = sameBBDetected;
        person.alert = alert;
        person.remove = remove;
        person.posture = posture;
        person.bad_prediction = bad_prediction;
        person.good_prediction = good_prediction;
        return person;
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
        if (!exactlySame(rect)) {
            sameBBDetected = 0;
        }
        this.rect = rect;
        this.lastseenTime = 0;
    }

    public boolean exactlySame(Rect rect) {
        if (this.rect.x == rect.x && this.rect.y == rect.y && this.rect.height == rect.height && this.rect.width == rect.width) {
            return true;
        } else {
            return false;
        }
    }


    public int getID() {
        return id;
    }

    public void tick() {
        lastmoveTime += 1;
        lastseenTime += 1;
        if (bad_prediction > Parameters.badValue) {
            badCounter++;
        } else {
            badCounter = 0;
        }

        if (badCounter > badLimit) {
            alert = true;
        }

        if (bad_prediction < Parameters.badValue && lastmoveTime < badLimit) {
            alert = false;
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
        int badLimit;

        public Persons(int movementMaximum, int movementMinimum, int movementTime) {
            this.movementMaximum = movementMaximum;
            this.movementMinimum = movementMinimum;
            this.badLimit = movementTime;
        }

        public Person addPerson(Rect rect) {
            Person p = familiarPerson(rect);
            if (p != null) {
                p.editPerson(rect);
                return p;
            } else {
                Person new_person = new Person(rect, movementMaximum, movementMinimum, badLimit);
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
