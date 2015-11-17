package edu.cmu.gardner.neural_nets

import breeze.linalg.{DenseMatrix,DenseVector}

class NeuralNetwork {
  var _base_layer: NetworkLayer = null
  private var _learning_rate = .1
  private var _num_iters = 100

  def addLayer(input_size: Int, output_size: Int, f: NetworkFunction) {
    val new_layer = new NetworkLayer(input_size, output_size, f)
    if (_base_layer == null) {
      _base_layer = new_layer
    } else {
      _base_layer.addLayer(new_layer)
    }
  }

  def trainWeights(data: Array[(DenseVector[Double], DenseVector[Double])]) {
    for (iter <- 0 until _num_iters) {
      for (i <- 0 until data.length) {
        _base_layer.feedForward(data(i)._1)
        _base_layer.propagateBackward(data(i)._2)
        _base_layer.updateWeights(_learning_rate)
      }
    }
  }

  def computeValue(input: DenseVector[Double]) = _base_layer.feedForward(input)
}
