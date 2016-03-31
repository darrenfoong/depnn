package uk.ac.cam.cl.depnn.io;

import java.util.ArrayList;
import java.util.HashSet;

import org.canova.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import uk.ac.cam.cl.depnn.embeddings.Embeddings;
import uk.ac.cam.cl.depnn.nn.NeuralNetwork;

public class TransDependency extends NNType {
	private int sigmoidScaleFactor = 20;

	@Override
	public int getNumProperties() {
		return 5;
	}

	private String stripCategory(String category) {
		return category.replaceAll("\\[.*?\\]", "");
	}

	@Override
	public NNType makeRecord(ArrayList<Writable> record, boolean hardLabels, HashSet<String> catLexicon, HashSet<String> slotLexicon, HashSet<String> distLexicon, HashSet<String> posLexicon) {
		TransDependency result = new TransDependency();

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

		if ( !category.equals("(S\\NP)/NP") || !slot.equals("2") ) {
			// TODO handle null return
			return null;
		}

		distLexicon.add(distance);
		posLexicon.add(headPos);
		posLexicon.add(dependentPos);

		result.add(head);
		result.add(dependent);
		result.add(distance);
		result.add(headPos);
		result.add(dependentPos);

		return result;
	}

	@Override
	public INDArray makeVector(NeuralNetwork<? extends NNType> network) {
		String head = this.get(0);
		String dependent = this.get(1);
		String distance = this.get(2);
		String headPos = this.get(3);
		String dependentPos = this.get(4);

		INDArray headVector = network.getWordVector(head);
		INDArray dependentVector = network.getWordVector(dependent);

		INDArray distanceVector = network.distEmbeddings.getINDArray(distance);
		INDArray headPosVector = network.posEmbeddings.getINDArray(headPos);
		INDArray dependentPosVector= network.posEmbeddings.getINDArray(dependentPos);

		return Nd4j.concat(1, headVector,
							dependentVector,
							distanceVector,
							headPosVector,
							dependentPosVector);
	}

	@Override
	public void updateEmbeddings(INDArray errors, int w2vLayerSize, Embeddings catEmbeddings, Embeddings slotEmbeddings, Embeddings distEmbeddings, Embeddings posEmbeddings) {
		distEmbeddings.updateEmbedding(this.get(2), errors, 2 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(3), errors, 3 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(4), errors , 4 * w2vLayerSize);
	}
}
