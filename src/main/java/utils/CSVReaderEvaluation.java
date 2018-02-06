package utils;

import java.io.*;
import java.util.ArrayList;

public class CSVReaderEvaluation extends AbstractCSVReader<EvaluationValue> {
    public CSVReaderEvaluation(File file, String csvSplitBy) {
        super(file, csvSplitBy);
    }


    @Override
    public void read() {
        list = new ArrayList<>();
        try {
            br = new BufferedReader(new FileReader(file));
            while ((line = br.readLine()) != null) {
                String[] row = line.split(csvSplitBy);
                EvaluationValue evaluationValue = new EvaluationValue();
                evaluationValue.setFrame_number(row[0]);
                if (row.length > 1) {
                    evaluationValue.setBb_values(row[1] + "," + row[2] + "," + row[3] + "," + row[4]);
                    evaluationValue.setPerson_there(row[5]);
                    evaluationValue.setPosture(row[6]);
                    evaluationValue.setIsMoving(row[7]);
                    evaluationValue.setStatus(row[8]);
                }
                list.add(evaluationValue);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    @Override
    public ArrayList<EvaluationValue> getValues() {
        return list;
    }
}
