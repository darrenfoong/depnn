package uk.ac.cam.cl.depnn;

import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import joptsimple.OptionException;
import joptsimple.OptionParser;
import joptsimple.OptionSet;
import uk.ac.cam.cl.depnn.io.Dependency;
import uk.ac.cam.cl.depnn.io.Feature;
import uk.ac.cam.cl.depnn.io.NNType;
import uk.ac.cam.cl.depnn.io.Params;
import uk.ac.cam.cl.depnn.io.TransDependency;
import uk.ac.cam.cl.depnn.nn.NeuralNetwork;

public class TrainNetwork {
	public static void main(String[] args) {
		OptionParser optionParser = Params.getTrainNetworkOptionParser();
		OptionSet options = null;

		try {
			options = optionParser.parse(args);
			if ( options.has("help") ) {
				optionParser.printHelpOn(System.out);
				return;
			}
		} catch ( OptionException e ) {
			System.err.println(e.getMessage());
			return;
		} catch ( IOException e ) {
			System.err.println(e);
			return;
		}

		String sentencesFile = (String) options.valueOf("sentencesFile");
		String trainDir = (String) options.valueOf("trainDir");
		String modelDir = (String) options.valueOf("modelDir");
		String logFile = (String) options.valueOf("log");
		String prevModelFile = null;

		if ( options.has("prevModel") ) {
			prevModelFile = (String) options.valueOf("prevModel");
		}

		int w2vSeed = (Integer) options.valueOf("w2vSeed");
		int w2vIterations = (Integer) options.valueOf("w2vIterations");
		int w2vBatchSize = (Integer) options.valueOf("w2vBatchSize");
		int w2vLayerSize = (Integer) options.valueOf("w2vLayerSize");
		int w2vWindowSize = (Integer) options.valueOf("w2vWindowSize");
		int w2vMinWordFreq = (Integer) options.valueOf("w2vMinWordFreq");
		int w2vNegativeSample = (Integer) options.valueOf("w2vNegativeSample");
		double w2vLearningRate = (Double) options.valueOf("w2vLearningRate");

		String nnType = (String) options.valueOf("nnType");
		int nnEpochs = (Integer) options.valueOf("nnEpochs");
		int nnSeed = (Integer) options.valueOf("nnSeed");
		int nnIterations = (Integer) options.valueOf("nnIterations");
		int nnBatchSize = (Integer) options.valueOf("nnBatchSize");
		int nnHiddenLayerSize = (Integer) options.valueOf("nnHiddenLayerSize");
		double nnLearningRate = (Double) options.valueOf("nnLearningRate");
		double nnL2Reg = (Double) options.valueOf("nnL2Reg");
		double nnDropout = (Double) options.valueOf("nnDropout");
		double nnEmbedRandomRange = (Double) options.valueOf("nnEmbedRandomRange");
		boolean nnHardLabels = (Boolean) options.valueOf("nnHardLabels");

		int maxNumBatch = (Integer) options.valueOf("maxNumBatch");

		System.setProperty("logLevel", options.has("verbose") ? "trace" : "info");
		System.setProperty("logFile", logFile);
		final Logger logger = LogManager.getLogger(TrainNetwork.class);

		logger.info(Params.printOptions(options));

		try {
			NeuralNetwork<? extends NNType> network = null;

			logger.info("Initializing network");

			if ( prevModelFile == null ) {
				switch ( nnType ) {
					case "dep":
						network = new NeuralNetwork<Dependency>(
								w2vSeed,
								w2vIterations,
								w2vBatchSize,
								w2vLayerSize,
								w2vWindowSize,
								w2vMinWordFreq,
								w2vNegativeSample,
								w2vLearningRate,
								nnEpochs,
								nnSeed,
								nnIterations,
								nnBatchSize,
								nnHiddenLayerSize,
								nnLearningRate,
								nnL2Reg,
								nnDropout,
								nnEmbedRandomRange,
								nnHardLabels,
								maxNumBatch,
								new Dependency());
						break;
					case "transdep":
						network = new NeuralNetwork<TransDependency>(
								w2vSeed,
								w2vIterations,
								w2vBatchSize,
								w2vLayerSize,
								w2vWindowSize,
								w2vMinWordFreq,
								w2vNegativeSample,
								w2vLearningRate,
								nnEpochs,
								nnSeed,
								nnIterations,
								nnBatchSize,
								nnHiddenLayerSize,
								nnLearningRate,
								nnL2Reg,
								nnDropout,
								nnEmbedRandomRange,
								nnHardLabels,
								maxNumBatch,
								new TransDependency());
						break;
					case "feature":
						network = new NeuralNetwork<Feature>(
								w2vSeed,
								w2vIterations,
								w2vBatchSize,
								w2vLayerSize,
								w2vWindowSize,
								w2vMinWordFreq,
								w2vNegativeSample,
								w2vLearningRate,
								nnEpochs,
								nnSeed,
								nnIterations,
								nnBatchSize,
								nnHiddenLayerSize,
								nnLearningRate,
								nnL2Reg,
								nnDropout,
								nnEmbedRandomRange,
								nnHardLabels,
								maxNumBatch,
								new Feature());
						break;
					default:
						throw new IllegalArgumentException("Invalid nnType");
				}
			} else {
				logger.info("Using previous word2vec model: " + prevModelFile);
				switch ( nnType ) {
					case "dep":
						network = new NeuralNetwork<Dependency>(
								prevModelFile,
								nnEpochs,
								nnSeed,
								nnIterations,
								nnBatchSize,
								nnHiddenLayerSize,
								nnLearningRate,
								nnL2Reg,
								nnDropout,
								nnEmbedRandomRange,
								nnHardLabels,
								maxNumBatch,
								new Dependency());
						break;
					case "transdep":
						network = new NeuralNetwork<TransDependency>(
								prevModelFile,
								nnEpochs,
								nnSeed,
								nnIterations,
								nnBatchSize,
								nnHiddenLayerSize,
								nnLearningRate,
								nnL2Reg,
								nnDropout,
								nnEmbedRandomRange,
								nnHardLabels,
								maxNumBatch,
								new TransDependency());
						break;
					case "feature":
						network = new NeuralNetwork<Feature>(
								prevModelFile,
								nnEpochs,
								nnSeed,
								nnIterations,
								nnBatchSize,
								nnHiddenLayerSize,
								nnLearningRate,
								nnL2Reg,
								nnDropout,
								nnEmbedRandomRange,
								nnHardLabels,
								maxNumBatch,
								new Feature());
						break;
					default:
						throw new IllegalArgumentException("Invalid nnType");
				}
			}

			logger.info("Network initialized");

			if ( prevModelFile == null ) {
				network.trainWord2Vec(sentencesFile);
				network.serializeWord2Vec(modelDir + "/word2vec.model");
			}

			network.trainNetwork(trainDir, modelDir);

			network.serialize(modelDir);
		} catch ( Exception e ) {
			logger.error("Exception", e);
		}
	}
}
