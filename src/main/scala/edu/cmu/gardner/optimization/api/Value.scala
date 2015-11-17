package edu.cmu.gardner.optimization.api

import breeze.linalg.DenseVector

// TODO(matt): Is this really necessary?  The reason I did this in the first place was to allow for
// relatively easy extension to multi-valued functions.  I suppose if this doesn't make the rest of
// the code too ugly, keeping it this way should allow for flexibility in the future...

sealed trait Value {
  def +(that: Value): Value
  def -(that: Value): Value
  def *(scalar: Double): Value
  def /(scalar: Double): Value
}

case class Scalar(x: Double) extends Value {
  override def +(that: Value) = Scalar(x + that.asInstanceOf[Scalar].x)
  override def -(that: Value) = Scalar(x - that.asInstanceOf[Scalar].x)
  override def *(scalar: Double) = Scalar(x * scalar)
  override def /(scalar: Double) = Scalar(x / scalar)
}
