package uk.ac.cam.cl.depnn;

import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import joptsimple.OptionException;
import joptsimple.OptionParser;
import joptsimple.OptionSet;
import uk.ac.cam.cl.depnn.io.Params;

public class TestNetwork {
	public static void main(String[] args) {
		OptionParser optionParser = Params.getTestNetworkOptionParser();
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

		String testDir = (String) options.valueOf("testDir");
		String modelDir = (String) options.valueOf("modelDir");
		String logFile = (String) options.valueOf("log");
		String modelFile = null;

		if ( options.has("prevModel") ) {
			modelFile = (String) options.valueOf("prevModel");
		} else {
			modelFile = modelDir + "/word2vec.model";
		}

		String configJsonFile = modelDir + "/config.json";
		String coefficientsFile = modelDir + "/coeffs";
		String catEmbeddingsFile = modelDir + "/cat.emb";
		String slotEmbeddingsFile = modelDir + "/slot.emb";
		String distEmbeddingsFile = modelDir + "/dist.emb";
		String posEmbeddingsFile = modelDir + "/pos.emb";

		System.setProperty("logLevel", options.has("verbose") ? "trace" : "info");
		System.setProperty("logFile", logFile);
		final Logger logger = LogManager.getLogger(TrainNetwork.class);

		logger.info(Params.printOptions(options));

		try {
			logger.info("Initializing network");
			DependencyNeuralNetwork depnn = new DependencyNeuralNetwork(modelFile,
												configJsonFile,
												coefficientsFile,
												catEmbeddingsFile,
												slotEmbeddingsFile,
												distEmbeddingsFile,
												posEmbeddingsFile);
			logger.info("Network initialized");

			depnn.testNetwork(testDir);
		} catch ( Exception e ) {
			logger.error("Exception", e);
		}
	}
}
