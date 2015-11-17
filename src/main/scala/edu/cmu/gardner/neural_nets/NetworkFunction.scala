package edu.cmu.gardner.neural_nets

import breeze.linalg.DenseVector
import breeze.numerics.tanh

abstract class NetworkFunction {
  def value(x: DenseVector[Double]): DenseVector[Double]
  def name: String
  def derivative(
    x: DenseVector[Double],
    value_at_x: DenseVector[Double],
    correct_output: DenseVector[Double] = null): DenseVector[Double]
}

class Tanh extends NetworkFunction {
  override def name = "Tanh"
  override def value(x: DenseVector[Double]) = tanh(x)
  override def derivative(
    x: DenseVector[Double],
    value_at_x: DenseVector[Double],
    correct_output: DenseVector[Double] = null) = ((value_at_x :* value_at_x) * -1.0) + 1.0
}

class LinearOutputWithSquaredLoss extends NetworkFunction {
  override def name = "Linear Output with Squared Loss"
  override def value(x: DenseVector[Double]) = x
  override def derivative(
    x: DenseVector[Double],
    value_at_x: DenseVector[Double],
    correct_output: DenseVector[Double] = null) = value_at_x - correct_output
}
