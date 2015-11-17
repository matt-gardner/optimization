package edu.cmu.gardner.optimization.api

import breeze.linalg.SparseVector

trait Instance

trait SparseFeatureInstance[T <: AnyVal] extends Instance {
  def getFeatures(): SparseVector[T]
}
