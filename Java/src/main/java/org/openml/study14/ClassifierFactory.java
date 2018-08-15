package org.openml.study14;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.MultiSearch;
import weka.classifiers.meta.multisearch.RandomSearch;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.core.setupgenerator.AbstractParameter;
import weka.core.setupgenerator.MathParameter;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class ClassifierFactory {
	
	public static MultiSearch getRandomSearchSetup(Classifier baseclassifier, AbstractParameter[] searchParameters, Integer numIterations, Integer numExecutionSlots) throws Exception {
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
		randomSearchAlgorithm.setNumExecutionSlots(numExecutionSlots);
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
	
	public static Classifier getRandomSearchSVM(Integer numIterations, Integer numExecutionSlots) throws Exception {
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
		
		return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
	}
	
	public static Classifier getRandomSearchDecisionTree(Integer numIterations, Integer numExecutionSlots) throws Exception {
		J48 baseclassifier = new J48();
		
		MathParameter numFeatures = new MathParameter();
		numFeatures.setProperty("classifier.minNumObj");
		numFeatures.setBase(1);
		numFeatures.setExpression("I");
		numFeatures.setMin(1);
		numFeatures.setMax(20);
		numFeatures.setStep(1);

		MathParameter maxDepth = new MathParameter();
		maxDepth.setProperty("classifier.confidenceFactor");
		maxDepth.setBase(10);
		maxDepth.setExpression("pow(BASE,I)");
		maxDepth.setMin(-4);
		maxDepth.setMax(-1);
		maxDepth.setStep(1);
		
		AbstractParameter[] searchParameters = {numFeatures, maxDepth};
		return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
	}
	
	public static Classifier getRandomSearchLogistic(Integer numIterations, Integer numExecutionSlots) throws Exception {
		Logistic baseclassifier = new Logistic();

		MathParameter ridge = new MathParameter();
		ridge.setProperty("classifier.ridge");
		ridge.setBase(2);
		ridge.setExpression("pow(BASE,I)");
		ridge.setMin(-12);
		ridge.setMax(12);
		ridge.setStep(1);
		
		AbstractParameter[] searchParameters = {ridge};
		return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
	}
	
	public static Classifier getRandomSearchKnn(Integer numIterations, Integer numExecutionSlots) throws Exception {
		IBk baseclassifier = new IBk();
		
		MathParameter numNeighbours = new MathParameter();
		numNeighbours.setProperty("classifier.KNN");
		numNeighbours.setBase(1);
		numNeighbours.setExpression("I");
		numNeighbours.setMin(1);
		numNeighbours.setMax(50);
		numNeighbours.setStep(1);
		
		AbstractParameter[] searchParameters = {numNeighbours};
		return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
	}
		
	
	public static MultiSearch getRandomSearchGB(Integer numIterations, Integer numExecutionSlots) throws Exception {
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

		return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
	}

}
