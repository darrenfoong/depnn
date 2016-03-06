package uk.ac.cam.cl.depnn;

import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import joptsimple.OptionException;
import joptsimple.OptionParser;
import joptsimple.OptionSet;
import uk.ac.cam.cl.depnn.io.Params;

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
		String dependenciesDir = (String) options.valueOf("dependenciesDir");
		String modelDir = (String) options.valueOf("modelDir");
		String logFile = (String) options.valueOf("log");
		String prevModelFile = null;

		if ( options.has("prevModel") ) {
			prevModelFile = (String) options.valueOf("prevModel");
		}

		String modelFile = modelDir + "/word2vec.model";
		String configJsonFile = modelDir + "/config.json";
		String coefficientsFile = modelDir + "/coeffs";
		String catEmbeddingsFile = modelDir + "/cat.emb";
		String slotEmbeddingsFile = modelDir + "/slot.emb";
		String distEmbeddingsFile = modelDir + "/dist.emb";
		String posEmbeddingsFile = modelDir + "/pos.emb";

		int w2vSeed = (Integer) options.valueOf("w2vSeed");
		int w2vIterations = (Integer) options.valueOf("w2vIterations");
		int w2vBatchSize = (Integer) options.valueOf("w2vBatchSize");
		int w2vLayerSize = (Integer) options.valueOf("w2vLayerSize");
		int w2vWindowSize = (Integer) options.valueOf("w2vWindowSize");
		int w2vMinWordFreq = (Integer) options.valueOf("w2vMinWordFreq");
		int w2vNegativeSample = (Integer) options.valueOf("w2vNegativeSample");
		double w2vLearningRate = (Double) options.valueOf("w2vLearningRate");

		int nnSeed = (Integer) options.valueOf("nnSeed");
		int nnIterations = (Integer) options.valueOf("nnIterations");
		int nnBatchSize = (Integer) options.valueOf("nnBatchSize");
		int nnHiddenLayerSize = (Integer) options.valueOf("nnHiddenLayerSize");
		double nnLearningRate = (Double) options.valueOf("nnLearningRate");
		double nnL1Reg = (Double) options.valueOf("nnL1Reg");
		double nnL2Reg = (Double) options.valueOf("nnL2Reg");
		double nnDropout = (Double) options.valueOf("nnDropout");
		double nnEmbedRandomRange = (Double) options.valueOf("nnEmbedRandomRange");

		int maxNumBatch = (Integer) options.valueOf("maxNumBatch");

		System.setProperty("logLevel", options.has("verbose") ? "trace" : "info");
		System.setProperty("logFile", logFile);
		final Logger logger = LogManager.getLogger(TrainNetwork.class);

		logger.info(Params.printOptions(options));

		try {
			DependencyNeuralNetwork depnn;

			logger.info("Initializing network");

			if ( prevModelFile == null ) {
				depnn = new DependencyNeuralNetwork(
													w2vSeed,
													w2vIterations,
													w2vBatchSize,
													w2vLayerSize,
													w2vWindowSize,
													w2vMinWordFreq,
													w2vNegativeSample,
													w2vLearningRate,
													nnSeed,
													nnIterations,
													nnBatchSize,
													nnHiddenLayerSize,
													nnLearningRate,
													nnL1Reg,
													nnL2Reg,
													nnDropout,
													nnEmbedRandomRange,
													maxNumBatch);
			} else {
				logger.info("Using previous word2vec model: " + prevModelFile);
				depnn = new DependencyNeuralNetwork(
													prevModelFile,
													nnSeed,
													nnIterations,
													nnBatchSize,
													nnHiddenLayerSize,
													nnLearningRate,
													nnL1Reg,
													nnL2Reg,
													nnDropout,
													nnEmbedRandomRange,
													maxNumBatch);
			}

			logger.info("Network initialized");

			if ( prevModelFile == null ) {
				depnn.trainWord2Vec(sentencesFile);
				logger.info("Serializing word2vec to " + modelFile);
				depnn.serializeWord2Vec(modelFile);
			}

			depnn.trainNetwork(dependenciesDir);

			logger.info("Serializing network to " + configJsonFile + ", " + coefficientsFile);
			depnn.serializeNetwork(configJsonFile, coefficientsFile);

			logger.info("Serializing embeddings to " + catEmbeddingsFile + ", "
													 + slotEmbeddingsFile + ", "
													 + distEmbeddingsFile + ", "
													 + posEmbeddingsFile);
			depnn.serializeEmbeddings(catEmbeddingsFile, slotEmbeddingsFile, distEmbeddingsFile, posEmbeddingsFile);
		} catch ( Exception e ) {
			logger.error("Exception", e);
		}
	}
}
