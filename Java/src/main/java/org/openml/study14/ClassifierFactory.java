package org.openml.study14;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.MultiSearch;
import weka.classifiers.meta.multisearch.RandomSearch;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.setupgenerator.AbstractParameter;
import weka.core.setupgenerator.ListParameter;
import weka.core.setupgenerator.MLPLayersParameter;
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
		
		if (numIterations == null) {
			return baseclassifier;
		} else {
			return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
		}
	}

	
	public static Classifier getRandomSearchRandomForest(Integer numIterations, Integer numExecutionSlots) throws Exception {
		RandomForest baseclassifier = new RandomForest();
		
		MathParameter numFeatures = new MathParameter();
		numFeatures.setProperty("classifier.numFeatures");
		numFeatures.setBase(1);
		numFeatures.setExpression("I");
		numFeatures.setMin(0.1);
		numFeatures.setMax(0.9);
		numFeatures.setStep(0.1);
		
		AbstractParameter[] searchParameters = {numFeatures};
		
		if (numIterations == null) {
			return baseclassifier;
		} else {
			return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
		}
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
		
		if (numIterations == null) {
			return baseclassifier;
		} else {
			return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
		}
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
		
		if (numIterations == null) {
			return baseclassifier;
		} else {
			return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
		}
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
		
		if (numIterations == null) {
			return baseclassifier;
		} else {
			return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
		}
	}
		
	
	public static Classifier getRandomSearchGB(Integer numIterations, Integer numExecutionSlots) throws Exception {
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

		if (numIterations == null) {
			return baseclassifier;
		} else {
			return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
		}
	}
	

	public static Classifier getRandomSearchNeuralNetwork(Integer numIterations, Integer numExecutionSlots) throws Exception {
		MultilayerPerceptron baseclassifier = new MultilayerPerceptron();

		MLPLayersParameter hiddenlayers = new MLPLayersParameter();
		hiddenlayers.setProperty("classifier.hiddenLayers");
		hiddenlayers.setMinLayers(1);
		hiddenlayers.setMaxLayers(2);
		hiddenlayers.setMinLayerSize(32);
		hiddenlayers.setMaxLayerSize(512);

		MathParameter learningRate = new MathParameter();
		learningRate.setProperty("classifier.learningRate");
		learningRate.setBase(10);
		learningRate.setExpression("pow(BASE,I)");
		learningRate.setMin(-5);
		learningRate.setMax(0);
		learningRate.setStep(1);

		ListParameter decay = new ListParameter();
		decay.setProperty("classifier.decay");
		decay.setList("false true");
		

		MathParameter epochs = new MathParameter();
		epochs.setProperty("classifier.trainingTime");
		epochs.setBase(1);
		epochs.setExpression("I");
		epochs.setMin(2);
		epochs.setMax(50);
		epochs.setStep(1);

		MathParameter momentum = new MathParameter();
		momentum.setProperty("classifier.momentum");
		momentum.setBase(1);
		momentum.setExpression("I");
		momentum.setMin(0.1);
		momentum.setMax(0.9);
		momentum.setStep(0.1);
		
		AbstractParameter[] searchParameters = {hiddenlayers, learningRate, decay, epochs, momentum};
		
		if (numIterations == null) {
			return baseclassifier;
		} else {
			return getRandomSearchSetup(baseclassifier, searchParameters, numIterations, numExecutionSlots);
		}
	}
	
	public static FilteredClassifier getRandomSearchNaiveBayes(Integer numIterations, Integer numExecutionSlots) throws Exception {
		NaiveBayes nb = new NaiveBayes();
		
		Filter[] filter = new Filter[3];
		filter[0] = new ReplaceMissingValues();
		filter[1] = new RemoveUseless();
		filter[2] = new Normalize();
		
		MultiFilter multifilter = new MultiFilter();
		multifilter.setFilters(filter);
		
		FilteredClassifier classifier = new FilteredClassifier();
		classifier.setFilter(multifilter);
		classifier.setClassifier(nb);
		
		return classifier;
	}
}
