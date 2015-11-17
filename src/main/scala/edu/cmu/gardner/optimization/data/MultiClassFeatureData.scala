package edu.cmu.gardner.optimization.data

import scala.util.Random

import breeze.linalg.DenseVector

object MultiClassBinaryFeatureData {
  val random = new Random()

  def sample_multinomial(probs: Seq[Double]) = {
    var p = random.nextDouble()
    var i = 0
    while (i < probs.size && p > probs(i)) {
      p -= probs(i)
      i += 1
    }
    i
  }

  def generate(
      num_instances: Int,
      class_probs: Seq[Double],
      params: Seq[Seq[Double]]) = {
    (1 to num_instances).par.map(i => {
      val cls = sample_multinomial(class_probs)
      val x = params.map(p => {
        val prob = p(cls)
        if (random.nextDouble() < prob) {
          1.0
        } else {
          0.0
        }
      }) ++ Seq(1.0)
      (new DenseVector[Double](x.toArray), cls)
    }).seq
  }
}
