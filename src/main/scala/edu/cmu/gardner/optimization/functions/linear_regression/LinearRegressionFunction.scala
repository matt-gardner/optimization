package edu.cmu.gardner.optimization.functions.linear_regression

import breeze.linalg.DenseVector

import edu.cmu.gardner.optimization.api.Instance
import edu.cmu.gardner.optimization.api.Function
import edu.cmu.gardner.optimization.api.Parameters

// We're assuming in the code that follows that x already has a bias term added.
case class LRInstance(x: DenseVector[Double], y: Double) extends Instance

case class LRParameters(theta: DenseVector[Double]) extends Parameters {
  override def asVector = theta
  override def update(grad: DenseVector[Double]) = LRParameters(theta + grad)
  override def vectorSize = theta.size
  override def fromVector(vector: DenseVector[Double]) = LRParameters(vector)
}

class LinearRegression extends Function {
  override def value(_params: Parameters, _instance: Instance): Double = {
    val params = _params.asInstanceOf[LRParameters]
    val instance = _instance.asInstanceOf[LRInstance]
    params.theta dot instance.x
  }

  override def gradient(_params: Parameters, _instance: Instance): DenseVector[Double] = {
    val instance = _instance.asInstanceOf[LRInstance]
    instance.x
  }
}

class LinearRegressionLossFunction extends Function {
  val linearRegression = new LinearRegression

  override def value(_params: Parameters, _instance: Instance): Double = {
    val instance = _instance.asInstanceOf[LRInstance]
    val diff = linearRegression.value(_params, _instance) - instance.y
    .5 * diff * diff
  }

  override def gradient(_params: Parameters, _instance: Instance): DenseVector[Double] = {
    val instance = _instance.asInstanceOf[LRInstance]
    val diff = linearRegression.value(_params, _instance) - instance.y
    instance.x * diff
  }
}
