package edu.cmu.gardner.optimization.optimizers.gradient_descent

import breeze.linalg.DenseVector
import breeze.linalg.sum
import breeze.numerics.abs

import edu.cmu.gardner.optimization.api.Instance
import edu.cmu.gardner.optimization.api.Function
import edu.cmu.gardner.optimization.api.Parameters

// TODO(matt): These should be put inside a params JValue
class GradientDescentOptimizer(
    learning_rate: Double,
    stopping_criterion: Double,
    l2_weight: Double = 0,
    l1_weight: Double = 0) {

  def regularization_term(params: Parameters) = {
    var value = 0.0
    if (l2_weight > 0) {
      value += (params.asVector dot params.asVector) * l2_weight
    }
    if (l1_weight > 0) {
      value += sum(abs(params.asVector))
    }
    value
  }

  def optimize(function: Function, initial_params: Parameters, instances: Seq[Instance]): Parameters = {
    var prev_value = Double.MaxValue
    var current_params = initial_params
    var value = function.batchValue(current_params, instances) + regularization_term(current_params)
    println("\nCurrent params: " + current_params)
    println("Current value: " + value)
    while (prev_value - value > stopping_criterion) {
      val gradient = function.batchGradient(current_params, instances)
      var update = -gradient
      if (l2_weight > 0) {
        update -= current_params.asVector * l2_weight
      }
      update *= learning_rate
      current_params = current_params.update(update)
      if (l1_weight > 0) {
        val vector = current_params.asVector
        val threshold = l1_weight * learning_rate
        for (i <- (0 until current_params.vectorSize)) {
          if (vector(i) > 0) {
            vector(i) = Math.max(0, vector(i) - threshold)
          } else {
            vector(i) = Math.min(0, vector(i) + threshold)
          }
        }
        current_params = current_params.fromVector(vector)
      }
      prev_value = value
      value = function.batchValue(current_params, instances) + regularization_term(current_params)
      println("\nCurrent params: " + current_params)
      println("Current value: " + value)
    }
    current_params
  }
}

object GradientDescentOptimizer {

  def main(args: Array[String]) {
    //linear_regression_test()
    logistic_regression_test()
  }

  def linear_regression_test() {
    import edu.cmu.gardner.optimization.functions.linear_regression.LRInstance
    import edu.cmu.gardner.optimization.functions.linear_regression.LRParameters
    import edu.cmu.gardner.optimization.data.LinearDataWithGaussianNoise
    import edu.cmu.gardner.optimization.functions.linear_regression.LinearRegressionLossFunction

    val num_d = 100000
    val instances = LinearDataWithGaussianNoise.generate(num_d, Seq(.3, .2, .8), .4, -1, 1, .1).map(
      i => LRInstance(i._1, i._2))
    val params = LRParameters(DenseVector.zeros[Double](4))
    val optimizer = new GradientDescentOptimizer(1.0 / num_d, 0.01, 0.0)
    println("Optimizing linear regression function")
    val optimized = optimizer.optimize(new LinearRegressionLossFunction(), params, instances)
  }

  def logistic_regression_test() {
    import edu.cmu.gardner.optimization.functions.logistic_regression.LRInstance
    import edu.cmu.gardner.optimization.functions.logistic_regression.LRParameters
    import edu.cmu.gardner.optimization.data.MultiClassBinaryFeatureData
    import edu.cmu.gardner.optimization.functions.logistic_regression.LogisticRegressionLossFunction

    val num_d = 100000
    val instances = MultiClassBinaryFeatureData.generate(num_d, Seq(.3, .7),
      Seq(Seq(.9, .2), Seq(.5, .5), Seq(.3, .9), Seq(0.0, .5))).map(
      i => LRInstance(i._1, i._2 == 0))
    val params = LRParameters(DenseVector.zeros[Double](5))
    val optimizer = new GradientDescentOptimizer(1.0 / num_d, 10, 10.0, 200.0)
    println("Optimizing logistic regression function")
    val optimized = optimizer.optimize(new LogisticRegressionLossFunction(), params, instances)
  }
}
