package utils;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import net.sourceforge.jFuzzyLogic.plot.JFuzzyChart;
import net.sourceforge.jFuzzyLogic.rule.Variable;

public class Test {
    public static void main(String[] args) {
        String filename = "tipper.fcl";
        FIS fis = FIS.load(filename, true);

        if (fis == null) {
            System.err.println("Can't load file: '" + filename + "'");
            System.exit(1);
        }

        // Get default function block
        FunctionBlock fb = fis.getFunctionBlock(null);

        // Set inputs
        fb.setVariable("posture", 1);
        fb.setVariable("time", 24);
        fb.setVariable("xposition", 320);
        fb.setVariable("yposition", 280);

        Variable posture = fb.getVariable("posture");
        posture.setValue(1);
        JFuzzyChart.get().chart(posture,true);

        Variable time = fb.getVariable("time");
        time.setValue(24);
        JFuzzyChart.get().chart(time,true);

        Variable xposition = fb.getVariable("xposition");
        xposition.setValue(300);
        JFuzzyChart.get().chart(xposition,true);

        Variable yposition = fb.getVariable("yposition");
        yposition.setValue(300);
        JFuzzyChart.get().chart(yposition,true);

        // Evaluate
        fb.evaluate();
        // Show output variable's chart
        Variable status = fb.getVariable("status");
        JFuzzyChart.get().chart(status,true);

        status.defuzzify();

        JFuzzyChart.get().chart(status, status.getDefuzzifier(), true);

        //fis.getVariable("tip").chartDefuzzifier(true);

        // Print ruleSet
        System.out.println(fb);
        System.out.println("Status: " + status.getValue() + " bad: " + status.getMembership("bad") + " good: " + status.getMembership("good"));

    }
}
