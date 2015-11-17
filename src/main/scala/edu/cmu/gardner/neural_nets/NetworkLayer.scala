package edu.cmu.gardner.neural_nets

import breeze.linalg.{DenseVector,DenseMatrix}

class NetworkLayer(input_size: Int, output_size: Int, function: NetworkFunction) {
  private var _next_layer: NetworkLayer = null
  var _weights = DenseMatrix.rand[Double](output_size, input_size)
  private var _last_input: DenseVector[Double] = null
  private var _last_activation: DenseVector[Double] = null
  private var _last_output: DenseVector[Double] = null
  private var _deltas: DenseVector[Double] = null
  private var _verbose = false

  def name = function.name + " (" + input_size + ", " + output_size + ")"

  def addLayer(layer: NetworkLayer) {
    if (_next_layer == null) {
      _next_layer = layer
    } else {
      _next_layer.addLayer(layer)
    }
  }

  def feedForward(input: DenseVector[Double]): DenseVector[Double] = {
    _last_input = input
    val activation = _weights * input
    _last_activation = activation
    val output = function.value(activation)
    _last_output = output
    if (_next_layer != null) {
      return _next_layer.feedForward(output)
    } else {
      return output
    }
  }

  def propagateBackward(correct_output: DenseVector[Double]): DenseVector[Double] = {
    if (_next_layer == null) {
      _deltas = function.derivative(_last_activation, _last_output, correct_output)
      if (_verbose) {
        println("Top-level error:")
        println("Predicted output:")
        println(_last_output)
        println("Correct output:")
        println(correct_output)
        println("Deltas:")
        println(_deltas)
        println()
      }
    } else {
      val next_layer_deltas = _next_layer.propagateBackward(correct_output)
      _deltas = function.derivative(_last_activation, _last_output) :* next_layer_deltas
    }
    if (_verbose) {
      println(name + " next_layer_deltas:")
      println(_weights.t * _deltas)
      println()
    }
    _weights.t * _deltas
  }

  def updateWeights(learning_rate: Double) {
    _weights :-= _deltas * _last_input.t * learning_rate
    if (_verbose) {
      println(name + " weights:")
      println(_weights)
      println()
    }
    if (_next_layer != null) {
      _next_layer.updateWeights(learning_rate)
    }
  }
}
