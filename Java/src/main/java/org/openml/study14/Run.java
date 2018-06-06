package org.openml.study14;

import java.util.Arrays;

import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.weka.algorithm.WekaConfig;
import org.openml.weka.experiment.RunOpenmlJob;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.MultiSearch;
import weka.classifiers.meta.multisearch.RandomSearch;
import weka.classifiers.trees.REPTree;
import weka.core.setupgenerator.AbstractParameter;
import weka.core.setupgenerator.MathParameter;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class Run {
	
	private static MultiSearch getRandomSearchSetup(Classifier baseclassifier, AbstractParameter[] searchParameters, Integer numIterations) throws Exception {
		Filter[] filter = new Filter[3];
		filter[0] = new ReplaceMissingValues();
		filter[1] = new RemoveUseless();
		filter[2] = new Normalize();
		
		MultiFilter multifilter = new MultiFilter();
		multifilter.setFilters(filter);
		
		FilteredClassifier classifier = new FilteredClassifier();
		classifier.setFilter(multifilter);
		classifier.setClassifier(baseclassifier);
		
		RandomSearch randomSearchAlgorithm = new RandomSearch();
		randomSearchAlgorithm.setNumIterations(numIterations);
		randomSearchAlgorithm.setSearchSpaceNumFolds(3);
		
		MultiSearch search = new MultiSearch();
		String[] evaluation = {"-E", "ACC"};
		search.setOptions(evaluation);
		search.setClassifier(classifier);
		search.setAlgorithm(randomSearchAlgorithm);
		search.setSearchParameters(searchParameters);
		
		return search;
	}
	
	private static Classifier getRandomSearchSVM(Integer numIterations) throws Exception {
		SMO baseclassifier = new SMO();
		baseclassifier.setKernel(new RBFKernel());
		
		MathParameter gamma = new MathParameter();
		gamma.setProperty("classifier.kernel.gamma");
		gamma.setBase(2);
		gamma.setExpression("pow(BASE,I)");
		gamma.setMin(-12);
		gamma.setMax(12);
		gamma.setStep(1);
		
		MathParameter complexity = new MathParameter();
		complexity.setProperty("classifier.c");
		complexity.setBase(2);
		complexity.setExpression("pow(BASE,I)");
		complexity.setMin(-12);
		complexity.setMax(12);
		complexity.setStep(1);
		
		AbstractParameter[] searchParameters = {gamma, complexity};
		
		return getRandomSearchSetup(baseclassifier, searchParameters, numIterations);
	}
	
	public static MultiSearch getRandomSearchGB(Integer numIterations) throws Exception {
		LogitBoost baseclassifier = new LogitBoost();
		baseclassifier.setClassifier(new REPTree());
		
		MathParameter numGBIterations = new MathParameter();
		numGBIterations.setProperty("classifier.numIterations");
		numGBIterations.setBase(1);
		numGBIterations.setExpression("I");
		numGBIterations.setMin(500);
		numGBIterations.setMax(10000);
		numGBIterations.setStep(1);
		

		MathParameter treeDepth = new MathParameter();
		treeDepth.setProperty("classifier.classifier.maxDepth");
		treeDepth.setBase(1);
		treeDepth.setExpression("I");
		treeDepth.setMin(1);
		treeDepth.setMax(5);
		treeDepth.setStep(1);

		MathParameter shrinkage = new MathParameter();
		shrinkage.setProperty("classifier.shrinkage");
		shrinkage.setBase(10);
		shrinkage.setExpression("pow(BASE,I)");
		shrinkage.setMin(-4);
		shrinkage.setMax(-1);
		shrinkage.setStep(1);
		
		AbstractParameter[] searchParameters = {numGBIterations, treeDepth, shrinkage};

		return getRandomSearchSetup(baseclassifier, searchParameters, numIterations);
	}
	
	public static void main(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		
		String function = args[0];
		Integer task_id = Integer.parseInt(args[1]);
		Integer numIterations = Integer.parseInt(args[2]);
		Classifier clf;
		
		if (function.equals("svm")) {
			clf = getRandomSearchSVM(numIterations);
		} else if (function.equals("gb")) {
			clf = getRandomSearchGB(numIterations);
		} else {
			throw new Exception("Unknown classifier option: " + function);
		}
		WekaConfig wekaconfig = new WekaConfig();
		
		RunOpenmlJob.executeTask(new OpenmlConnector(wekaconfig.getServer(), wekaconfig.getApiKey()), wekaconfig, task_id, clf);
		
	}
}
