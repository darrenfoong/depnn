package uk.ac.cam.cl.depnn;

import java.io.FileNotFoundException;
import java.io.IOException;

public class RunDependencyNeuralNetwork {
	public static void main(String[] args) {
		DependencyNeuralNetwork depnn = new DependencyNeuralNetwork();

		try {
			depnn.importData("");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
