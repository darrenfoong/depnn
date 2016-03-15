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

		String nnType = (String) options.valueOf("nnType");

		System.setProperty("logLevel", options.has("verbose") ? "trace" : "info");
		System.setProperty("logFile", logFile);
		final Logger logger = LogManager.getLogger(TrainNetwork.class);

		logger.info(Params.printOptions(options));

		try {
			NeuralNetwork<? extends NNType> depnn;

			logger.info("Initializing network");

			if ( nnType.equals("dep") ) {
				depnn = new NeuralNetwork<Dependency>(modelDir, new Dependency());
			} else {
				depnn = new NeuralNetwork<Feature>(modelDir, new Feature());
			}

			logger.info("Network initialized");

			depnn.testNetwork(testDir);
		} catch ( Exception e ) {
			logger.error("Exception", e);
		}
	}
}
