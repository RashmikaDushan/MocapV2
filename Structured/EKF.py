# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,too-many-instance-attributes, too-many-arguments


"""Copyright 2015 Roger R Labbe Jr."""

from __future__ import (absolute_import, division, unicode_literals)

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye
import scipy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z
from MathLib import quaternion_multiply_normalized
import quaternion


class ExtendedKalmanFilter(object):

    """ Implements an extended Kalman filter (EKF). You are responsible for
    implementing f and state variables; the defaults will  not give you a 
    functional filter.

    Parameters
    ----------

    dim_x : int


        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.

        This is used to set the default size of P, Q, and u

    dim_u : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_u would be 2.

    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        State estimate vector

    P : numpy.array(dim_x, dim_x)
        Covariance matrix

    x_predi : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_predi : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    R : numpy.array(dim_u, dim_u)
        Measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        Process noise matrix

    F : numpy.array()
        Jacobian of state transition function

    y : numpy.array
        Residual of the update step. Read only.

    K : numpy.array(dim_x, dim_u)
        Kalman gain of the update step. Read only.

    S :  numpy.array
        Systen uncertaintly projected to measurement space. Read only.

    z : ndarray
        Last measurement used in update(). Read only.

    log_likelihood : float
        log-likelihood of the last measurement. Read only.

    likelihood : float
        likelihood of last measurment. Read only.

        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.

    mahalanobis : float
        mahalanobis distance of the innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Read only.
    """

    def __init__(self, dim_x, dim_u):

        self.dim_x = dim_x
        self.dim_u = dim_u

        self.x = zeros((dim_x)) # state
        self.P = eye(dim_x)        # uncertainty covariance
        self.B = 0                 # control transition matrix
        self.F = zeros((dim_x,dim_x))       # Jacobian of f
        self.R = eye(dim_u)        # state uncertainty
        self.Q = eye(dim_x)        # process uncertainty
        self.y = zeros((dim_u, 1)) # residual

        z = np.array([None]*self.dim_u)
        self.z = reshape_z(z, self.dim_u, self.x.ndim)

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros(self.x.shape) # kalman gain
        self.y = zeros((dim_u, 1))
        self.S = np.zeros((dim_u, dim_u))   # system uncertainty
        self.SI = np.zeros((dim_u, dim_u))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # these will always be a copy of x,P after predict() is called
        self.x_predi = self.x.copy()
        self.P_predi = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        
    def f(self, x, dt, u):
        """
        State transition function for the EKF.

        1. Compute orientation update:
            q_{k+1} = q_k + 0.5 * Δt * q_k ⊗ [0; ω_m]
            q_{k+1} = normalize(q_{k+1})

        2. Compute rotation matrix R(q_k)

        3. Compute world acceleration:
            a_w = R(q_k) * a_m + g

        4. Update velocity:
            v_{k+1} = v_k + Δt * a_w

        5. Update position:
            p_{k+1} = p_k + Δt * v_k + 0.5 * Δt^2 * a_w
        """
        # Extract states
        # x_k = [q0, q1, q2, q3, x, y, z, vx, vy, vz, bwx, bwy, bwz, bax, bay, baz]
        # u = [ax, ay, az, wx, wy, wz, mx, my, mz]
        global a_w
        q = x[0:4]
        pos = x[4:7]
        vel = x[7:10]

        a = u[0:3] - x[10:13]  # Accelerometer measurement minus bias
        w = u[3:6] - x[13:16]  # Gyroscope measurement minus bias
        
        # Initialize new state
        x_new = np.zeros((self.dim_x))
        
        # 1. Compute orientation update:
        # q_{k+1} = q_k + 0.5 * Δt * q_k ⊗ [0; ω_m]
        x_new[0:4] = q + 0.5 * dt * quaternion_multiply_normalized(q, np.array([0, w[0], w[1], w[2]]))  # Quaternion update
        

        # 2. Compute rotation matrix R(q_k)
        R = quaternion.as_rotation_matrix(quaternion.from_float_array(x_new[0:4]))

        # 3. Compute world acceleration:
        # a_w = R(q_k) * a_m + g
        a_w = np.dot(R, a) + np.array([0, 0, -9.81])  # Gravity vector in world frame

        # 4. Update velocity:
        # v_{k+1} = v_k + Δt * a_w
        x_new[7:10] = vel + dt * a_w

        # 5. Update position:
        # p_{k+1} = p_k + Δt * v_k + 0.5 * Δt^2 * a_w
        x_new[4:7] = pos + dt * vel + 0.5 * dt**2 * a_w

        x_new[10:16] = x[10:16]  # Biases remain unchanged in this model
        
        return x_new
    
    def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
        """ Performs the predict/update innovation of the extended Kalman
        filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, only predict step is perfomed.

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, along with the
           optional arguments in args, and returns H.

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable.

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx after the required state
            variable.

        u : np.array or scalar
            optional control vector input to the filter.
        """
        #pylint: disable=too-many-locals

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if np.isscalar(z) and self.dim_u == 1:
            z = np.asarray([z], float)

        F = self.F
        B = self.B
        P = self.P
        Q = self.Q
        R = self.R
        x = self.x

        H = HJacobian(x, *args)

        # predict step
        x = self.f()
        P = dot(F, P).dot(F.T) + Q

        # save prior
        self.x_predi = np.copy(self.x)
        self.P_predi = np.copy(self.P)

        # update step
        PHT = dot(P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = linalg.inv(self.S)
        self.K = dot(PHT, self.SI)

        self.y = z - Hx(x, *hx_args)
        self.x = x + dot(self.K, self.y)

        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract):
        """ Performs the update innovation of the extended Kalman filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, posterior is not computed

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable. for robot localization you might need to pass in
            information about the map and time of day, so you might have
            `args=(map_data, time)`, where the signature of HCacobian will
            be `def HJacobian(x, map, t)`

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        if z is None:
            self.z = np.array([[None]*self.dim_u]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_u) * R

        if np.isscalar(z) and self.dim_u == 1:
            z = np.asarray([z], float)

        H = HJacobian(self.x, *args)

        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.K = PHT.dot(linalg.inv(self.S))

        hx = Hx(self.x, *hx_args)
        self.y = residual(z, hx)
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, u=0):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        self.predict_x(u)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        # save prior
        self.x_predi = np.copy(self.x)
        self.P_predi = np.copy(self.P)

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """

        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """
        Mahalanobis distance of innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

    def __repr__(self):
        return '\n'.join([
            'KalmanFilter object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_predi', self.x_predi),
            pretty_str('P_predi', self.P_predi),
            pretty_str('F', self.F),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('K', self.K),
            pretty_str('y', self.y),
            pretty_str('S', self.S),
            pretty_str('likelihood', self.likelihood),
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('mahalanobis', self.mahalanobis)
            ])

ekf = ExtendedKalmanFilter(dim_x=16, dim_u=9)
x_1 = np.zeros((16))  # [q0, q1, q2, q3, x, y, z, vx, vy, vz, bwx, bwy, bwz, bax, bay, baz]
x_1[0] = 1  # Initial quaternion
u_1 = np.array([0, 9.81, 0, 0.1, 0, 0, 0, 1, 0])
x_2 = ekf.f(x_1, 0.1, u_1)  # Predict next state
print("Predicted state:", x_2)