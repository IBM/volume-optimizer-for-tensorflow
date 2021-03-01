# Copyright (C) 2021, International Business Machines
# Corporation.  All Rights Reserved.

# This program is distributed under the terms of the
# Eclipse Public License - v 2.0



"""Volume optimizer for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.training import training_ops
import tensorflow as tf



class VOLUME(Optimizer):
    def __init__(self, learning_rate=1.e-1, alpha1=0.2, alpha2=0.2, eps=1.e-8,
                 use_locking=False, name="Volume"):
        super(VOLUME, self).__init__(use_locking, name)
        self._stp = learning_rate
        self._stpmax = learning_rate * 2
        self._stpmin = learning_rate * 0.2
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        self._eps = eps

        # Tensor versions of the constructor arguments, created in _prepare().
        self._stp_t = None
        self._stpmax_t = None
        self._stpmin_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._gy = None
        self._updated_stp = None
        self._pp = None
        self._niter = None
        self._alpha = None

    
    def _create_slots(self, var_list):
    
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._gy is None
        if not create_new and context.in_graph_mode():
            create_new = (self._gy.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._gy = variable_scope.variable(0.5,
                                                   name="gy",
                                                   trainable=False)
                self._updated_stp = variable_scope.variable(self._stp,
                                                            name="updated_stp",
                                                            trainable=False)
                self._pp = variable_scope.variable(0.,
                                                   name="pp",
                                                   trainable=False)
                self._niter = variable_scope.variable(0.,
                                                   name="niter",
                                                   trainable=False)
                self._alpha = variable_scope.variable(0.,
                                                   name="alpha",
                                                   trainable=False)
                
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _prepare(self):
        self._stp_t = ops.convert_to_tensor(self._stp, name="stp")
        self._stpmax_t = ops.convert_to_tensor(self._stpmax, name="stpmax")
        self._stpmin_t = ops.convert_to_tensor(self._stpmin, name="stpmin")
     
        
    def _apply_dense(self, grad, var):
        gy = math_ops.cast(self._gy, var.dtype.base_dtype)
        updated_stp = math_ops.cast(self._updated_stp, var.dtype.base_dtype)
        niter = math_ops.cast(self._niter, var.dtype.base_dtype)
        stpmax_t = math_ops.cast(self._stpmax_t, var.dtype.base_dtype)
        stpmin_t = math_ops.cast(self._stpmin_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha, var.dtype.base_dtype)
        alpha_t = tf.where( tf.less(niter, 500000), self._alpha1, self._alpha2)
    
        m = self.get_slot(var, "m")
        #
        pp=tf.reduce_sum(tf.multiply(m, grad, name=None))
        g0 = gy * (1.0 - alpha_t)
        g1 = g0 + alpha_t
        gy = tf.where( tf.greater(pp,0.0), g1, g0 )

        #
        updated_stp = tf.where( tf.greater(gy, 0.66), updated_stp * 1.01, updated_stp )
        updated_stp = tf.where( tf.less(gy, 0.33), updated_stp * 0.99, updated_stp )
        updated_stp = tf.where( tf.greater(updated_stp, stpmax_t), stpmax_t, updated_stp )
        updated_stp = tf.where( tf.less(updated_stp, stpmin_t), stpmin_t, updated_stp)
        
        
        m_t = state_ops.assign(m, m * (1. - alpha_t) + grad * alpha_t,
                               use_locking=self._use_locking)
        
        norma = math_ops.sqrt(tf.reduce_sum(tf.multiply(m_t, m_t, name=None)))
        
        lr = updated_stp / (norma + self._eps)
        
        var_update = state_ops.assign_sub(var,
                                          lr * m_t,
                                          use_locking=self._use_locking)
        niter = niter + 1
        
        update_pp = self._pp.assign(pp, use_locking=self._use_locking)
        update_gy = self._gy.assign(gy, use_locking=self._use_locking)
        update_stp_op = self._updated_stp.assign(updated_stp, use_locking=self._use_locking)   
        update_niter = self._niter.assign(niter, use_locking=self._use_locking)
        update_alpha = self._alpha.assign(alpha_t, use_locking=self._use_locking)
        
        
        return control_flow_ops.group(*[var_update, m_t, 
                                        update_pp,
                                        update_gy, update_stp_op,
                                        update_niter,
                                        update_alpha,
                                        debug_print_op])
        
      

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)
    
    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

