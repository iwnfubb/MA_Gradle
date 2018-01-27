package main;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import net.sourceforge.jFuzzyLogic.plot.JFuzzyChart;
import net.sourceforge.jFuzzyLogic.rule.Variable;
import utils.Utils;

public class TestFuzzy {
    public static void main(String[] args) {
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

        // Print ruleSet
        System.out.println("Status: " + status.getValue());
    }
}
