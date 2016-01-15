package uk.ac.cam.cl.depnn;

import java.io.FileNotFoundException;

public class RunDependencyNeuralNetwork {
	public static void main(String[] args) {
		DependencyNeuralNetwork depnn = new DependencyNeuralNetwork();

		try {
			depnn.trainWord2Vec("raw_sentences.txt");
			depnn.trainNetwork("deps");

			System.out.println(depnn.predict("drink", "lion"));
			System.out.println(depnn.predict("drink", "tiger"));
			System.out.println(depnn.predict("watch", "coffee"));
			System.out.println(depnn.predict("watch", "television"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
