package edu.cmu.gardner.optimization.api

import breeze.linalg.DenseVector

// Currently only supports single-valued functions.  That should be enough for the things I want to
// experiment with, though, as optimizing multi-valued functions is a little problematic, anyway...
// The best you can do is define some function over the multiple values and optimize that, or try
// to find some paretal optimal solution.  The first case is pretty easily handled by having a
// function that takes other functions as input.  The second case we'll punt on.
trait Function {
  def value(params: Parameters, instance: Instance): Double
  def batchValue(params: Parameters, instances: Seq[Instance]): Double = {
    instances.par.map(instance => value(params, instance)).sum
  }

  def gradient(params: Parameters, instance: Instance): DenseVector[Double]
  def batchGradient(params: Parameters, instances: Seq[Instance]): DenseVector[Double] = {
    val grad = DenseVector.zeros[Double](params.vectorSize)
    // TODO(matt): Maybe not the most efficient thing to do, as this probably creates lots of
    // objects...  The right thing to do is probably create a synchronized gradient object that
    // gets updated in place after each call to gradient().
    instances.par.map(instance => gradient(params, instance)).seq.foldLeft(grad)(_ + _)
  }

  def numericalGradient(params: Parameters, instance: Instance, epsilon: Double): DenseVector[Double] = {
    val partials = (0 until params.vectorSize).par.map(i => {
      val delta = DenseVector.zeros[Double](params.vectorSize)
      delta(i) = epsilon
      val params_plus = params.update(delta)
      // TODO(matt): currently doesn't handle other value types
      val value_plus = value(params_plus, instance)
      delta(i) = -epsilon
      val params_minus = params.update(delta)
      val value_minus = value(params_minus, instance)
      val gradient = ((value_plus - value_minus) / (2 * epsilon))
      (i, gradient)
    })
    val gradient = DenseVector.zeros[Double](params.vectorSize)
    partials.foreach(v => gradient(v._1) = v._2)
    gradient
  }
}
