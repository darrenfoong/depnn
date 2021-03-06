package uk.ac.cam.cl.depnn.io;

import java.util.ArrayList;
import java.util.HashSet;

import org.canova.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import uk.ac.cam.cl.depnn.embeddings.Embeddings;
import uk.ac.cam.cl.depnn.nn.NeuralNetwork;

public class LongDependency extends NNType {
	private int sigmoidScaleFactor = 20;

	@Override
	public int getNumProperties() {
		return 11;
	}

	private String stripCategory(String category) {
		return category.replaceAll("\\[.*?\\]", "");
	}

	@Override
	public NNType makeRecord(ArrayList<Writable> record, boolean hardLabels, HashSet<String> catLexicon, HashSet<String> slotLexicon, HashSet<String> distLexicon, HashSet<String> posLexicon) {
		LongDependency result = new LongDependency();

		String head = record.get(0).toString();
		String category = stripCategory(record.get(1).toString());
		String slot = record.get(2).toString();
		String dependent = record.get(3).toString();
		String distance = record.get(4).toString();
		String headPos = record.get(5).toString();
		String dependentPos = record.get(6).toString();
		String headLeftPos = record.get(7).toString();
		String headRightPos = record.get(8).toString();
		String dependentLeftPos = record.get(9).toString();
		String dependentRightPos = record.get(10).toString();

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
		posLexicon.add(headLeftPos);
		posLexicon.add(headRightPos);
		posLexicon.add(dependentLeftPos);
		posLexicon.add(dependentRightPos);

		result.add(head);
		result.add(category);
		result.add(slot);
		result.add(dependent);
		result.add(distance);
		result.add(headPos);
		result.add(dependentPos);
		result.add(headLeftPos);
		result.add(headRightPos);
		result.add(dependentLeftPos);
		result.add(dependentRightPos);

		return result;
	}

	@Override
	public INDArray makeVector(NeuralNetwork<? extends NNType> network) {
		// head category slot dependent distance head_pos dependent_pos value count
		String head = this.get(0);
		String category = stripCategory(this.get(1));
		String slot = this.get(2);
		String dependent = this.get(3);
		String distance = this.get(4);
		String headPos = this.get(5);
		String dependentPos = this.get(6);
		String headLeftPos = this.get(7);
		String headRightPos = this.get(8);
		String dependentLeftPos = this.get(9);
		String dependentRightPos = this.get(10);

		INDArray headVector = network.getWordVector(head);
		INDArray dependentVector = network.getWordVector(dependent);

		INDArray categoryVector = network.catEmbeddings.getINDArray(category);
		INDArray slotVector = network.slotEmbeddings.getINDArray(slot);
		INDArray distanceVector = network.distEmbeddings.getINDArray(distance);
		INDArray headPosVector = network.posEmbeddings.getINDArray(headPos);
		INDArray dependentPosVector= network.posEmbeddings.getINDArray(dependentPos);
		INDArray headLeftPosVector = network.posEmbeddings.getINDArray(headLeftPos);
		INDArray headRightPosVector = network.posEmbeddings.getINDArray(headRightPos);
		INDArray dependentLeftPosVector= network.posEmbeddings.getINDArray(dependentLeftPos);
		INDArray dependentRightPosVector= network.posEmbeddings.getINDArray(dependentRightPos);

		return Nd4j.concat(1, headVector,
							categoryVector,
							slotVector,
							dependentVector,
							distanceVector,
							headPosVector,
							dependentPosVector,
							headLeftPosVector,
							headRightPosVector,
							dependentLeftPosVector,
							dependentRightPosVector);
	}

	@Override
	public void updateEmbeddings(INDArray errors, int w2vLayerSize, Embeddings catEmbeddings, Embeddings slotEmbeddings, Embeddings distEmbeddings, Embeddings posEmbeddings) {
		catEmbeddings.updateEmbedding(this.get(1), errors, 1 * w2vLayerSize);
		slotEmbeddings.updateEmbedding(this.get(2), errors, 2 * w2vLayerSize);
		distEmbeddings.updateEmbedding(this.get(4), errors, 4 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(5), errors, 5 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(6), errors , 6 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(7), errors, 7 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(8), errors , 8 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(9), errors, 9 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(10), errors , 10 * w2vLayerSize);
	}
}
