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

		String trainDir = (String) options.valueOf("trainDir");
		String modelDir = (String) options.valueOf("modelDir");
		String logFile = (String) options.valueOf("log");
		String prevModelFile = (String) options.valueOf("prevModel");

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
				case "longdep":
					network = new NeuralNetwork<LongDependency>(
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
							new LongDependency());
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

			logger.info("Network initialized");

			network.trainNetwork(trainDir, modelDir);

			network.serialize(modelDir);
		} catch ( Exception e ) {
			logger.error("Exception", e);
		}
	}
}
