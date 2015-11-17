package edu.cmu.gardner.optimization.api

import breeze.linalg.DenseVector

trait Parameters {
  def vectorSize: Int
  def asVector: DenseVector[Double]
  def update(delta: DenseVector[Double]): Parameters
  // It's not the cleanest design to have this be an instance method instead of an object method,
  // but the optimizer code will only have access to instances.
  def fromVector(vector: DenseVector[Double]): Parameters
}
