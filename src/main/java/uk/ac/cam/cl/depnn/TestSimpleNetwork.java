package uk.ac.cam.cl.depnn;

import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import joptsimple.OptionException;
import joptsimple.OptionParser;
import joptsimple.OptionSet;
import uk.ac.cam.cl.depnn.io.Dependency;
import uk.ac.cam.cl.depnn.io.Feature;
import uk.ac.cam.cl.depnn.io.LongDependency;
import uk.ac.cam.cl.depnn.io.NNType;
import uk.ac.cam.cl.depnn.io.Params;
import uk.ac.cam.cl.depnn.io.TransDependency;
import uk.ac.cam.cl.depnn.nn.SimpleNeuralNetwork;

public class TestSimpleNetwork {
	public static void main(String[] args) {
		OptionParser optionParser = Params.getTestSimpleNetworkOptionParser();
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
		double nnPosThres = (Double) options.valueOf("nnPosThres");
		double nnNegThres = (Double) options.valueOf("nnNegThres");
		boolean precompute = (Boolean) options.valueOf("precompute");

		System.setProperty("logLevel", options.has("verbose") ? "trace" : "info");
		System.setProperty("logFile", logFile);
		final Logger logger = LogManager.getLogger(TestSimpleNetwork.class);

		logger.info(Params.printOptions(options));

		try {
			SimpleNeuralNetwork<? extends NNType> network = null;

			logger.info("Initializing network");

			switch ( nnType ) {
				case "dep":
					network = new SimpleNeuralNetwork<Dependency>(modelDir, precompute, new Dependency());
					break;
				case "longdep":
					network = new SimpleNeuralNetwork<LongDependency>(modelDir, precompute, new LongDependency());
					break;
				case "transdep":
					network = new SimpleNeuralNetwork<TransDependency>(modelDir, precompute, new TransDependency());
					break;
				case "feature":
					network = new SimpleNeuralNetwork<Feature>(modelDir, precompute, new Feature());
					break;
				default:
					throw new IllegalArgumentException("Invalid nnType");
			}

			logger.info("Network initialized");

			network.testNetwork(testDir, logFile, nnPosThres, nnNegThres);
		} catch ( Exception e ) {
			logger.error("Exception", e);
		}
	}
}
