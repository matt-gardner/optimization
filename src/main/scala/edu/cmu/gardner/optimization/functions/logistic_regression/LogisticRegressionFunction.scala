package edu.cmu.gardner.optimization.functions.logistic_regression

import breeze.linalg.DenseVector

import edu.cmu.gardner.optimization.api.Instance
import edu.cmu.gardner.optimization.api.Function
import edu.cmu.gardner.optimization.api.Parameters

// We're assuming in the code that follows that x already has a bias term added.
case class LRInstance(x: DenseVector[Double], y: Boolean) extends Instance

case class LRParameters(theta: DenseVector[Double]) extends Parameters {
  override def asVector = theta
  override def update(grad: DenseVector[Double]) = LRParameters(theta + grad)
  override def vectorSize = theta.size
  override def fromVector(vector: DenseVector[Double]) = LRParameters(vector)
}

class LogisticRegression extends Function {
  override def value(_params: Parameters, _instance: Instance): Double = {
    val params = _params.asInstanceOf[LRParameters]
    val instance = _instance.asInstanceOf[LRInstance]
    val dot = params.theta dot instance.x
    val exponent = Math.exp(-dot)
    1 / (1 + exponent)
  }

  override def gradient(_params: Parameters, _instance: Instance): DenseVector[Double] = {
    val instance = _instance.asInstanceOf[LRInstance]
    instance.x
  }
}

class LogisticRegressionLossFunction extends Function {
  val logisticRegression = new LogisticRegression

  override def value(_params: Parameters, _instance: Instance): Double = {
    val instance = _instance.asInstanceOf[LRInstance]
    val prob = logisticRegression.value(_params, _instance)
    if (instance.y) {
      -Math.log(prob)
    } else {
      -Math.log(1 - prob)
    }
  }

  override def gradient(_params: Parameters, _instance: Instance): DenseVector[Double] = {
    val instance = _instance.asInstanceOf[LRInstance]
    val prob = logisticRegression.value(_params, _instance)
    val y = if (instance.y) 1 else 0
    instance.x * (prob - y)
  }
}
