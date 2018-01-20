package algorithms;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import net.sourceforge.jFuzzyLogic.rule.Variable;
import utils.Utils;

public class FuzzyModel {

    FunctionBlock fb;

    public FuzzyModel() {
        String filename = Utils.PATH_TO_RESOURCES_FOLDER + "tipper.fcl";
        FIS fis = FIS.load(filename, true);

        if (fis == null) {
            System.err.println("Can't load file: '" + filename + "'");
            System.exit(1);
        }
        // Get default function block
        fb = fis.getFunctionBlock(null);
    }

    public String[] evaluate(int posture, double time, double xposition, double yposition) {
        Variable postureVariable = fb.getVariable("posture");
        postureVariable.setValue(posture);
        //JFuzzyChart.get().chart(postureVariable, true);

        Variable timeVariable = fb.getVariable("time");
        timeVariable.setValue(time);
        //JFuzzyChart.get().chart(timeVariable, true);

        Variable xpositionVariable = fb.getVariable("xposition");
        xpositionVariable.setValue(xposition);
        //JFuzzyChart.get().chart(xpositionVariable, true);

        Variable ypositionVariable = fb.getVariable("yposition");
        ypositionVariable.setValue(yposition);
        //JFuzzyChart.get().chart(ypositionVariable, true);

        // Evaluate
        fb.evaluate();
        // Show output variable's chart
        Variable status = fb.getVariable("status");
        //JFuzzyChart.get().chart(status, true);

        status.defuzzify();

        //JFuzzyChart.get().chart(status, status.getDefuzzifier(), true);

        // Print ruleSet
        String str = ("Status: " + status.getValue() + "\n"
                + " bad: " + status.getMembership("bad") + "\n"
                + " good: " + status.getMembership("good") + "\n");
        System.out.println(str);
        return new String[]{"Status: " + status.getValue(),
                " bad: " + status.getMembership("bad"),
                " good: " + status.getMembership("good")};
    }
}