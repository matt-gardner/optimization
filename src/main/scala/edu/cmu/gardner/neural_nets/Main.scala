package edu.cmu.gardner.neural_nets

import breeze.linalg.{DenseVector,DenseMatrix}
import breeze.linalg.linspace
import breeze.plot.Figure
import breeze.plot.plot

import scala.collection.mutable.ArrayBuffer

object Main {
  def getTrainingDataForFunction(f: Double => Double) = {
    val domain_start = 0.0
    val domain_end = 1.0
    val domain_steps = 100
    val step_size = (domain_end - domain_start) / domain_steps
    var current = domain_start
    var data = new ArrayBuffer[(DenseVector[Double], DenseVector[Double])]()
    while (current <= domain_end) {
      val example = (new DenseVector[Double](Array(current)), new DenseVector[Double](Array(f(current))))
      data += example
      current += step_size
    }
    util.Random.shuffle(data).toArray
  }


  def main(args: Array[String]) {
    var make_plot = false
    val network = new NeuralNetwork()
    network.addLayer(1, 10, new Tanh())
    network.addLayer(10, 1, new LinearOutputWithSquaredLoss())
    val f = (x: Double) => x * x
    val data = getTrainingDataForFunction(f)
    println("Training weights")
    network.trainWeights(data)
    println("Tanh weights:")
    println(network._base_layer._weights)
    if (args.length > 0) make_plot = true

    if (make_plot) {
      val fig = Figure()
      val p = fig.subplot(0)
      val x = linspace(0.0,1.0)
      val y = x.map(v => network.computeValue(new DenseVector[Double](Array(v))).valueAt(0))
      println(x)
      println(y)
      p += plot(x, y, '.')
      p += plot(x, x :^ 2.0)
      p.xlabel = "x axis"
      p.ylabel = "y axis"
      fig.saveas("lines.png")
    }
  }
}
