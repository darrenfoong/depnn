package uk.ac.cam.cl.depnn.nn;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import uk.ac.cam.cl.depnn.io.NNType;
import uk.ac.cam.cl.depnn.io.PrecomputesManager;

public class SimpleMultiLayerNetwork<T extends NNType> {
	private int INPUT_LAYER_SIZE;
	private int HIDDEN_LAYER_SIZE;
	private int OUTPUT_LAYER_SIZE;

	private INDArray w_h;
	private INDArray w_out;
	private INDArray b_h;
	private INDArray b_out;

	private final static Logger logger = LogManager.getLogger(SimpleMultiLayerNetwork.class);

	public INDArray getMatrix(int offset, int size) {
		// return w_h.get(NDArrayIndex.point(i * size), NDArrayIndex.interval(i * size, (i+1) * size));
		// ugly, but certainly works
		INDArray res = new NDArray(size, HIDDEN_LAYER_SIZE);

		for ( int i = 0; i < size; i++ ) {
			res.putRow(i, w_h.getRow((offset * size) + i));
		}

		return res;
	}

	public SimpleMultiLayerNetwork(String coefficientsFile, int inputLayerSize, int hiddenLayerSize, int outputLayerSize) throws IOException {
		INPUT_LAYER_SIZE = inputLayerSize;
		HIDDEN_LAYER_SIZE = hiddenLayerSize;
		OUTPUT_LAYER_SIZE = outputLayerSize;

		DataInputStream dis = new DataInputStream(new FileInputStream(coefficientsFile));
		INDArray newParams = Nd4j.read(dis);

		int idx = 0;
		int range = 0;

		range = INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE;
		w_h = newParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, range + idx));
		w_h = w_h.reshape(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE).transposei();
		idx += range;

		range = 1 * HIDDEN_LAYER_SIZE;
		b_h = newParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, range + idx));
		b_h = b_h.reshape(HIDDEN_LAYER_SIZE, 1).transposei();
		idx += range;

		range = HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE;
		w_out = newParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, range + idx));
		w_out = w_out.reshape(OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE).transposei();
		idx += range;

		range = 1 * OUTPUT_LAYER_SIZE;
		b_out = newParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, range + idx));
		b_out = b_out.reshape(OUTPUT_LAYER_SIZE, 1).transposei();
		idx += range;
	}

	public INDArray relui(INDArray inputs) {
		for ( int i = 0; i < inputs.rows(); i++ ) {
			for ( int j = 0; j < inputs.columns(); j++ ) {
				if ( inputs.getDouble(i, j) < 0 ) {
					inputs.put(i, j, 0);
				}
			}
		}

		return inputs;
	}

	public INDArray softmaxi(INDArray inputs) {
		for ( int i = 0; i < inputs.rows(); i++ ) {
			double sum = 0.0;

			for ( int j = 0; j < inputs.columns(); j++ ) {
				sum += Math.exp(inputs.getDouble(i, j));
			}

			for ( int j = 0; j < inputs.columns(); j++ ) {
				inputs.put(i, j, Math.exp(inputs.getDouble(i, j)) / sum);
			}
		}

		return inputs;
	}

	public INDArray output(INDArray inputs, boolean training) {
		INDArray hidden_layer = relui(inputs.mmul(w_h).addiRowVector(b_h));
		return softmaxi(hidden_layer.mmul(w_out).addiRowVector(b_out));
	}

	public INDArray output(List<T> list, PrecomputesManager<T> manager) {
		INDArray pre_hidden_layer = new NDArray(list.size(), HIDDEN_LAYER_SIZE);

		for ( int i = 0; i < list.size(); i++ ) {
			INDArray sub_hidden_layer = Nd4j.zeros(HIDDEN_LAYER_SIZE);
			NNType example = list.get(i);

			for ( int j = 0; j < manager.getNumPrecomputes(); j++ ) {
				sub_hidden_layer.addi(manager.getPrecomputes(j).getINDArray(example.get(j)));
			}

			pre_hidden_layer.putRow(i, sub_hidden_layer);
		}

		INDArray hidden_layer = relui(pre_hidden_layer.addiRowVector(b_h));
		return softmaxi(hidden_layer.mmul(w_out).addiRowVector(b_out));
	}
}