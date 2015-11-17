package edu.cmu.gardner.optimization.optimizers.gradient_descent

import breeze.linalg.DenseVector
import breeze.linalg.sum
import breeze.numerics.abs

import edu.cmu.gardner.optimization.api.Instance
import edu.cmu.gardner.optimization.api.Function
import edu.cmu.gardner.optimization.api.Parameters

import scala.util.Random

class StochasticGradientDescentOptimizer(
    initial_learning_rate: Double,
    alpha: Double,
    iterations: Int,
    l2_weight: Double = 0,
    l1_weight: Double = 0) {
  val random = new Random

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
    var learning_rate = initial_learning_rate
    var current_params = initial_params
    for (iter <- 1 to iterations) {
      println("\nCurrent params: " + current_params)
      val value = function.batchValue(current_params, instances) + regularization_term(current_params)
      println("Current value: " + value)
      println()
      for (instance <- random.shuffle(instances)) {
        val gradient = function.gradient(current_params, instance)
        var update = -gradient
        if (l2_weight > 0) {
          update -= current_params.asVector * l2_weight
        }
        update *= learning_rate
        current_params = current_params.update(update)
        if (l1_weight > 0) {
          // This is _slow_ for large feature spaces!  I need to write a special case of this
          // optimizer for large sparse feature spaces.
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
        learning_rate *= alpha
      }
    }
    current_params
  }
}

object StochasticGradientDescentOptimizer {

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
    val optimizer = new StochasticGradientDescentOptimizer(0.5, .99999, 5, 0.0)
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
    val optimizer = new StochasticGradientDescentOptimizer(0.5, .99999, 25, 0.01, 0.1)
    println("Optimizing logistic regression function")
    val optimized = optimizer.optimize(new LogisticRegressionLossFunction(), params, instances)
  }
}
