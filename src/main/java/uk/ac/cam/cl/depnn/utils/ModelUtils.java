package uk.ac.cam.cl.depnn.utils;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ModelUtils {
	private ModelUtils() {
	}

	public static void saveModelAndParameters(MultiLayerNetwork net, File confPath, String paramPath) throws IOException {
		// save parameters
		DataOutputStream dos = new DataOutputStream(new FileOutputStream(paramPath));
		Nd4j.write(net.params(), dos);
		dos.flush();
		dos.close();

		// save model configuration
		FileUtils.write(confPath, net.getLayerWiseConfigurations().toJson());
	}

	public static MultiLayerNetwork loadModelAndParameters(File confPath, String paramPath) throws IOException {
		// load parameters
		MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(confPath));
		DataInputStream dis = new DataInputStream(new FileInputStream(paramPath));
		INDArray newParams = Nd4j.read(dis);
		dis.close();

		// load model configuration
		MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
		savedNetwork.init();
		savedNetwork.setParams(newParams);

		return savedNetwork;
	}

	public static void saveLayerParameters(INDArray param, String paramPath) throws IOException {
		// save parameters for each layer
		DataOutputStream dos = new DataOutputStream(new FileOutputStream(paramPath));
		Nd4j.write(param, dos);
		dos.flush();
		dos.close();
	}

	public static Layer loadLayerParameters(Layer layer, String paramPath) throws IOException {
		// load parameters for each layer
		DataInputStream dis = new DataInputStream(new FileInputStream(paramPath));
		INDArray param = Nd4j.read(dis);
		dis.close();
		layer.setParams(param);
		return layer;
	}

	public static void saveParameters(MultiLayerNetwork model, int[] layerIds, Map<Integer, String> paramPaths) throws IOException {
		Layer layer;

		for (int layerId : layerIds) {
			layer = model.getLayer(layerId);
			if (!layer.paramTable().isEmpty()) {
				ModelUtils.saveLayerParameters(layer.params(), paramPaths.get(layerId));
			}
		}
	}

	public static void saveParameters(MultiLayerNetwork model, String[] layerIds, Map<String, String> paramPaths) throws IOException {
		Layer layer;

		for (String layerId : layerIds) {
			layer = model.getLayer(layerId);
			if (!layer.paramTable().isEmpty()) {
				ModelUtils.saveLayerParameters(layer.params(), paramPaths.get(layerId));
			}
		}
	}

	public static MultiLayerNetwork loadParameters(MultiLayerNetwork model, int[] layerIds, Map<Integer, String> paramPaths) throws IOException {
		Layer layer;

		for (int layerId : layerIds) {
			layer = model.getLayer(layerId);
			loadLayerParameters(layer, paramPaths.get(layerId));
		}

		return model;
	}

	public static MultiLayerNetwork loadParameters(MultiLayerNetwork model, String[] layerIds, Map<String, String> paramPaths) throws IOException {
		Layer layer;

		for (String layerId : layerIds) {
			layer = model.getLayer(layerId);
			loadLayerParameters(layer, paramPaths.get(layerId));
		}

		return model;
	}
}
