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

		String modelFile = modelDir + "/word2vec.model";
		String configJsonFile = modelDir + "/config.json";
		String coefficientsFile = modelDir + "/coeffs";
		String catEmbeddingsFile = modelDir + "/cat.emb";
		String slotEmbeddingsFile = modelDir + "/slot.emb";
		String distEmbeddingsFile = modelDir + "/dist.emb";
		String posEmbeddingsFile = modelDir + "/pos.emb";

		int w2vMinWordFreq = (Integer) options.valueOf("w2vMinWordFreq");
		int w2vIterations = (Integer) options.valueOf("w2vIterations");
		int w2vLayerSize = (Integer) options.valueOf("w2vLayerSize");
		int w2vSeed = (Integer) options.valueOf("w2vSeed");
		int w2vWindowSize = (Integer) options.valueOf("w2vWindowSize");

		int nnNumProperties = (Integer) options.valueOf("nnNumProperties");
		int nnBatchSize = (Integer) options.valueOf("nnBatchSize");
		int nnIterations = (Integer) options.valueOf("nnIterations");
		int nnHiddenLayerSize = (Integer) options.valueOf("nnHiddenLayerSize");
		int nnSeed = (Integer) options.valueOf("nnSeed");
		double nnLearningRate = (Double) options.valueOf("nnLearningRate");
		double nnL1Reg = (Double) options.valueOf("nnL1Reg");
		double nnL2Reg = (Double) options.valueOf("nnL2Reg");
		double nnDropout = (Double) options.valueOf("nnDropout");
		double nnEmbedRandomRange = (Double) options.valueOf("nnEmbedRandomRange");

		System.setProperty("logLevel", options.has("verbose") ? "trace" : "info");
		System.setProperty("logFile", logFile);
		final Logger logger = LogManager.getLogger(TrainNetwork.class);

		logger.info(Params.printOptions(options));

		DependencyNeuralNetwork depnn = new DependencyNeuralNetwork(
											w2vMinWordFreq,
											w2vIterations,
											w2vLayerSize,
											w2vSeed,
											w2vWindowSize,
											nnNumProperties,
											nnBatchSize,
											nnIterations,
											nnHiddenLayerSize,
											nnSeed,
											nnLearningRate,
											nnL1Reg,
											nnL2Reg,
											nnDropout,
											nnEmbedRandomRange);

		try {
			depnn.trainWord2Vec(sentencesFile);
			depnn.trainNetwork(dependenciesDir);

			depnn.serializeWord2Vec(modelFile);
			depnn.serializeNetwork(configJsonFile, coefficientsFile);
			depnn.serializeEmbeddings(catEmbeddingsFile, slotEmbeddingsFile, distEmbeddingsFile, posEmbeddingsFile);
		} catch ( Exception e ) {
			logger.info(e);
		}
	}
}
