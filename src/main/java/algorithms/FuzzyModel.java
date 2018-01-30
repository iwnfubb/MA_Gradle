package algorithms;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import net.sourceforge.jFuzzyLogic.rule.Variable;
import utils.Utils;

public class FuzzyModel {

    FunctionBlock fb;

    public FuzzyModel() {
        String filename = Utils.PATH_TO_RESOURCES_FOLDER + "sleep.fcl";
        FIS fis = FIS.load(filename, true);

        if (fis == null) {
            System.err.println("Can't load file: '" + filename + "'");
            System.exit(1);
        }
        // Get default function block
        fb = fis.getFunctionBlock(null);
    }

    public String[] string_evaluate(int posture, double time, double xposition, double yposition) {
        double[] evaluation = double_evaluate(posture, time, xposition, yposition);
        String str = ("Status: " + evaluation[0] + "\n"
                + " bad: " + evaluation[1] + "\n"
                + " good: " + evaluation[2] + "\n");
        System.out.println(str);
        return new String[]{"Status: " + evaluation[0],
                " bad: " + evaluation[1],
                " good: " + evaluation[2]};
    }

    public double[] double_evaluate(int posture, double time, double xposition, double yposition) {
        Variable postureVariable = fb.getVariable("posture");
        postureVariable.setValue(posture);

        Variable timeVariable = fb.getVariable("time");
        timeVariable.setValue(time);

        Variable xpositionVariable = fb.getVariable("xposition");
        xpositionVariable.setValue(xposition);

        Variable ypositionVariable = fb.getVariable("yposition");
        ypositionVariable.setValue(yposition);

        // Evaluate
        fb.evaluate();
        // Show output variable's chart
        Variable status = fb.getVariable("status");

        status.defuzzify();

        // Print ruleSet
        String str = ("Status: " + status.getValue() + "\n"
                + " bad: " + status.getMembership("bad") + "\n"
                + " good: " + status.getMembership("good") + "\n");
        System.out.println(str);
        return new double[]{status.getValue(),
                status.getMembership("bad"),
                status.getMembership("good")};
    }
}