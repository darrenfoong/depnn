package uk.ac.cam.cl.depnn;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class RunDependencyNeuralNetwork {
	public static void main(String[] args) {
		final Logger logger = LogManager.getLogger(RunDependencyNeuralNetwork.class);

		DependencyNeuralNetwork depnn = new DependencyNeuralNetwork();

		// depnn.trainWord2Vec("raw_sentences.txt");
		// depnn.trainNetwork("deps");
	}
}
