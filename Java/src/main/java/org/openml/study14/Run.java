package org.openml.study14;

import java.util.Arrays;

import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.weka.algorithm.WekaConfig;
import org.openml.weka.experiment.RunOpenmlJob;

import weka.classifiers.Classifier;

public class Run {
	
	
	
	public static void main(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		
		String function = args[0];
		Integer task_id = Integer.parseInt(args[1]);
		Integer numIterations = Integer.parseInt(args[2]);
		Integer numExecutionSlots = Integer.parseInt(args[3]);
		Classifier clf;
		
		if (function.equals("svm")) {
			clf = ClassifierFactory.getRandomSearchSVM(numIterations, numExecutionSlots);
		} else if (function.equals("gb")) {
			clf = ClassifierFactory.getRandomSearchGB(numIterations, numExecutionSlots);
		} else if (function.equals("dt")) {
			clf = ClassifierFactory.getRandomSearchDecisionTree(numIterations, numExecutionSlots);
		} else if (function.equals("knn")) {
			clf = ClassifierFactory.getRandomSearchKnn(numIterations, numExecutionSlots);
		} else if (function.equals("logistic")) {
			clf = ClassifierFactory.getRandomSearchLogistic(numIterations, numExecutionSlots);
		} else {
			throw new Exception("Unknown classifier option: " + function);
		}
		WekaConfig wekaconfig = new WekaConfig();
		
		RunOpenmlJob.executeTask(new OpenmlConnector(wekaconfig.getServer(), wekaconfig.getApiKey()), wekaconfig, task_id, clf);
		
	}
}
