package uk.ac.cam.cl.depnn.io;

import java.util.ArrayList;
import java.util.HashSet;

import org.canova.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import uk.ac.cam.cl.depnn.embeddings.Embeddings;
import uk.ac.cam.cl.depnn.nn.NeuralNetwork;

public class Dependency extends NNType {
	private int sigmoidScaleFactor = 20;

	@Override
	public int getNumProperties() {
		return 7;
	}

	public String stripCategory(String category) {
		return category.replaceAll("\\[.*?\\]", "");
	}

	@Override
	public NNType makeRecord(ArrayList<Writable> record, boolean hardLabels, HashSet<String> catLexicon, HashSet<String> slotLexicon, HashSet<String> distLexicon, HashSet<String> posLexicon) {
		Dependency result = new Dependency();

		String head = record.get(0).toString();
		String category = stripCategory(record.get(1).toString());
		String slot = record.get(2).toString();
		String dependent = record.get(3).toString();
		String distance = record.get(4).toString();
		String headPos = record.get(5).toString();
		String dependentPos = record.get(6).toString();

		String valueString;

		if ( hardLabels ) {
			valueString = record.get(11).toString();
			result.value = Double.parseDouble(valueString);
		} else {
			valueString = record.get(12).toString();
			result.value = Math.tanh(Double.parseDouble(valueString) / sigmoidScaleFactor);
		}

		catLexicon.add(category);
		slotLexicon.add(slot);
		distLexicon.add(distance);
		posLexicon.add(headPos);
		posLexicon.add(dependentPos);

		result.add(head);
		result.add(category);
		result.add(slot);
		result.add(dependent);
		result.add(distance);
		result.add(headPos);
		result.add(dependentPos);

		return result;
	}

	@Override
	public INDArray makeVector(NeuralNetwork<? extends NNType> network) {
		// head category slot dependent distance head_pos dependent_pos value count
		INDArray headVector;
		INDArray categoryVector;
		INDArray slotVector;
		INDArray dependentVector;
		INDArray distanceVector;
		INDArray headPosVector;
		INDArray dependentPosVector;

		if ( preloadList.isEmpty() ) {
			headVector = network.getWordVector(this.get(0));
			categoryVector = network.catEmbeddings.getINDArray(this.get(1));
			slotVector = network.slotEmbeddings.getINDArray(this.get(2));
			dependentVector = network.getWordVector(this.get(3));
			distanceVector = network.distEmbeddings.getINDArray(this.get(4));
			headPosVector = network.posEmbeddings.getINDArray(this.get(5));
			dependentPosVector = network.posEmbeddings.getINDArray(this.get(6));
		} else {
			headVector = preloadList.get(0);
			categoryVector = network.catEmbeddings.getINDArray(this.get(0));
			slotVector = network.slotEmbeddings.getINDArray(this.get(1));
			dependentVector = preloadList.get(1);
			distanceVector = network.distEmbeddings.getINDArray(this.get(2));
			headPosVector = preloadList.get(2);
			dependentPosVector = preloadList.get(3);
		}

		return Nd4j.concat(1, headVector,
							categoryVector,
							slotVector,
							dependentVector,
							distanceVector,
							headPosVector,
							dependentPosVector);
	}

	@Override
	public void updateEmbeddings(INDArray errors, int w2vLayerSize, Embeddings catEmbeddings, Embeddings slotEmbeddings, Embeddings distEmbeddings, Embeddings posEmbeddings) {
		catEmbeddings.updateEmbedding(this.get(1), errors, 1 * w2vLayerSize);
		slotEmbeddings.updateEmbedding(this.get(2), errors, 2 * w2vLayerSize);
		distEmbeddings.updateEmbedding(this.get(4), errors, 4 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(5), errors, 5 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(6), errors , 6 * w2vLayerSize);
	}
}
