package uk.ac.cam.cl.depnn.io;

import java.util.ArrayList;
import java.util.HashSet;

import org.canova.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import uk.ac.cam.cl.depnn.NeuralNetwork;
import uk.ac.cam.cl.depnn.embeddings.Embeddings;

public class Dependency extends NNType {
	private int sigmoidScaleFactor = 20;

	@Override
	public NNType makeRecord(ArrayList<Writable> record, boolean hardLabels, HashSet<String> catLexicon, HashSet<String> slotLexicon, HashSet<String> distLexicon, HashSet<String> posLexicon) {
		Dependency result = new Dependency();

		String head = record.get(0).toString();
		String category = record.get(1).toString();
		String slot = record.get(2).toString();
		String dependent = record.get(3).toString();
		String distance = record.get(4).toString();
		String headPos = record.get(5).toString();
		String dependentPos = record.get(6).toString();

		String valueString;

		if ( hardLabels ) {
			valueString = record.get(7).toString();
			result.value = Double.parseDouble(valueString);
		} else {
			valueString = record.get(8).toString();
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
	public INDArray makeVector(NeuralNetwork depnn) {
		// head category slot dependent distance head_pos dependent_pos value count
		String head = this.get(0);
		String category = this.get(1);
		String slot = this.get(2);
		String dependent = this.get(3);
		String distance = this.get(4);
		String headPos = this.get(5);
		String dependentPos = this.get(6);

		INDArray headVector = depnn.getWordVector(head);
		INDArray dependentVector = depnn.getWordVector(dependent);

		INDArray categoryVector = depnn.catEmbeddings.getINDArray(category);
		INDArray slotVector = depnn.slotEmbeddings.getINDArray(slot);
		INDArray distanceVector = depnn.distEmbeddings.getINDArray(distance);
		INDArray headPosVector = depnn.posEmbeddings.getINDArray(headPos);
		INDArray dependentPosVector= depnn.posEmbeddings.getINDArray(dependentPos);

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
		catEmbeddings.addEmbedding(this.get(1), errors, 1 * w2vLayerSize);
		slotEmbeddings.addEmbedding(this.get(2), errors, 2 * w2vLayerSize);
		distEmbeddings.addEmbedding(this.get(4), errors, 4 * w2vLayerSize);
		posEmbeddings.addEmbedding(this.get(5), errors, 5 * w2vLayerSize);
		posEmbeddings.addEmbedding(this.get(6), errors , 6 * w2vLayerSize);
	}
}
