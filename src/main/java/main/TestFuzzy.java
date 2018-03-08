package main;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import net.sourceforge.jFuzzyLogic.plot.JFuzzyChart;
import net.sourceforge.jFuzzyLogic.rule.Variable;
import utils.Utils;

public class TestFuzzy {
    public static void main_(String[] args) {
        String filename = Utils.PATH_TO_RESOURCES_FOLDER + "example.fcl";
        FIS fis = FIS.load(filename, true);

        if (fis == null) {
            System.err.println("Can't load file: '" + filename + "'");
            System.exit(1);
        }

        // Get default function block
        FunctionBlock fb = fis.getFunctionBlock(null);

        // Set inputs
        Variable posture = fb.getVariable("temperature");
        posture.setValue(23);
        JFuzzyChart.get().chart(posture, true);

        // Evaluate
        fb.evaluate();
        // Show output variable's chart
        Variable status = fb.getVariable("thermostat");
        JFuzzyChart.get().chart(status, true);

        status.defuzzify();

        JFuzzyChart.get().chart(status, status.getDefuzzifier(), true);

        // Print ruleSetdie
        System.out.println("Status: " + status.getValue());
    }

    public static void main(String[] args) {
        String filename = Utils.PATH_TO_RESOURCES_FOLDER + "sleep.fcl";
        FIS fis = FIS.load(filename, true);

        if (fis == null) {
            System.err.println("Can't load file: '" + filename + "'");
            System.exit(1);
        }

        Variable postureVariable = fis.getVariable("posture");
        postureVariable.setValue(1);
        JFuzzyChart.get().chart(postureVariable, true);


        Variable timeVariable = fis.getVariable("time");
        timeVariable.setValue(14);
        JFuzzyChart.get().chart(timeVariable, true);

        Variable xpositionVariable = fis.getVariable("xposition");
        xpositionVariable.setValue(574);
        JFuzzyChart.get().chart(xpositionVariable, true);

        Variable ypositionVariable = fis.getVariable("yposition");
        ypositionVariable.setValue(424);
        JFuzzyChart.get().chart(ypositionVariable, true);

        // Evaluate
        fis.evaluate();
        // Show output variable's chart
        Variable status = fis.getVariable("status");
        JFuzzyChart.get().chart(status, true);
        status.defuzzify();

        // Print ruleSet
        String str = ("Status: " + status.getValue() + "\n"
                + " bad: " + status.getMembership("bad") + "\n"
                + " good: " + status.getMembership("good") + "\n");
        System.out.println(str);
        JFuzzyChart.get().chart(status, status.getDefuzzifier(), true);
    }
}
