package edu.cmu.gardner.optimization.api

trait Optimizer {
  def optimize(function: Function, initial_params: Parameters, instances: Seq[Instance]): Parameters
}
