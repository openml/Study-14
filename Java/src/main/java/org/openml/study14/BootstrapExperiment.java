package org.openml.study14;

import java.util.Arrays;

import org.apache.commons.lang3.tuple.Pair;
import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.Run;
import org.openml.weka.algorithm.WekaConfig;
import org.openml.weka.experiment.RunOpenmlJob;

import weka.classifiers.Classifier;

public class BootstrapExperiment {

	private static final String VERSION = "20190227";

	public static void main(String[] args) throws Exception {
		System.out.println(VERSION);
		System.out.println(Arrays.toString(args));
		WekaConfig wekaconfig = new WekaConfig();
		OpenmlConnector openml = new OpenmlConnector(wekaconfig.getServer(), wekaconfig.getApiKey());
		System.out.println(wekaconfig);

		if (args.length < 4) {
			throw new Exception(
					"Program requires at least 4 CLI arguments: classifierName <str>, " + 
					"task_id <int>, numIterations <int>, numExecutionSlots <int>");
		}
		String classifierName = args[0];
		Integer taskId = Integer.parseInt(args[1]);
		Integer numIterations = Integer.parseInt(args[2]);
		Integer numExecutionSlots = Integer.parseInt(args[3]);
		Classifier clf;

		if (classifierName.equals("svm")) {
			clf = ClassifierFactory.getRandomSearchSVM(numIterations, numExecutionSlots);
		} else if (classifierName.equals("gb")) {
			clf = ClassifierFactory.getRandomSearchGB(numIterations, numExecutionSlots);
		} else if (classifierName.equals("dt")) {
			clf = ClassifierFactory.getRandomSearchDecisionTree(numIterations, numExecutionSlots);
		} else if (classifierName.equals("rf")) {
			clf = ClassifierFactory.getRandomSearchRandomForest(numIterations, numExecutionSlots);
		} else if (classifierName.equals("knn")) {
			clf = ClassifierFactory.getRandomSearchKnn(numIterations, numExecutionSlots);
		} else if (classifierName.equals("logistic")) {
			clf = ClassifierFactory.getRandomSearchLogistic(numIterations, numExecutionSlots);
		} else if (classifierName.equals("nn")) {
			clf = ClassifierFactory.getRandomSearchNeuralNetwork(numIterations, numExecutionSlots);
		} else if (classifierName.equals("nb")) {
			clf = ClassifierFactory.getRandomSearchNaiveBayes(numIterations, numExecutionSlots);
		} else {
			throw new Exception("Unknown classifier option: " + classifierName);
		}

		Pair<Integer, Run> result = RunOpenmlJob.executeTask(openml, wekaconfig, taskId, clf);
		Integer runId = result.getLeft();
		openml.runTag(runId, "ReproducibleBenchmark");
	}
}
