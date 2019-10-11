package org.openml.study14;

import java.util.Arrays;

import org.apache.commons.lang3.tuple.Pair;
import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.Run;
import org.openml.apiconnector.xml.Study;
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

		if (args.length < 3) {
			throw new Exception(
					"Program requires at least 3 CLI arguments: classifierName <str>, benchmark_suite <str>, task_idx <int>");
		}
		
		String classifierName = args[0];
		Study study = openml.studyGet(args[1], "tasks");
		Integer taskIdx = Integer.parseInt(args[2]);
		Integer taskId = study.getTasks()[taskIdx];
		Integer numIterations = null;
		Integer numExecutionSlots = null;
		Classifier clf;

		if (args.length > 3) {
			numIterations = Integer.parseInt(args[3]);
		}
		if (args.length > 4) {
			numExecutionSlots = Integer.parseInt(args[4]);
		}
		
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
		System.out.println("Run id: " + runId);
	}
}
