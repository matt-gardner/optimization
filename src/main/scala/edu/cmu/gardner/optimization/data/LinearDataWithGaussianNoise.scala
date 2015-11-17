package edu.cmu.gardner.optimization.data

import scala.util.Random

import breeze.linalg.DenseVector

object LinearDataWithGaussianNoise {
  val random = new Random()

  def generate(
      num_instances: Int,
      params: Seq[Double],
      bias: Double,
      x_min: Double,
      x_max: Double,
      variance: Double) = {
    (1 to num_instances).par.map(i => {
      val noise = random.nextGaussian * variance
      val x = params.map(p => random.nextDouble() * (x_max - x_min) + x_min) ++ Seq(1.0)
      val y = x.zip(params).map(z => z._1 * z._2).sum + bias + noise
      (new DenseVector[Double](x.toArray), y)
    }).seq
  }
}
