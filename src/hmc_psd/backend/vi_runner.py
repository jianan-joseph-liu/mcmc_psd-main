import time
import timeit
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

from ..logging import logger
from .analysis_data import AnalysisData
from .bayesian_model import BayesianModel

tfd = tfp.distributions
tfb = tfp.bijectors


class ViRunner:
    def __init__(
        self,
        x: np.ndarray,
        N_theta: int = 30,
        nchunks: int = 400,
        variation_factor: float = 0.0,
        fmax_for_analysis: float = None,
        fs: float = 2048,
        degree_fluctuate: float = None,
        init_params: List[tf.Tensor] = None,
        fmin_for_analysis: float = None,
        fmin_idx_extension: int = 0,
        fmax_idx_extension: int = 32,
        Nbw: float = 1.0,
    ):
        self.data = AnalysisData(
            x=x,
            nchunks=nchunks,
            fmax_for_analysis=fmax_for_analysis,
            fs=fs,
            N_theta=N_theta,
            N_delta=N_theta,  # N_theta == N_delta in all cases
            fmin_for_analysis=fmin_for_analysis,
            fmin_idx_extension=fmin_idx_extension,
            fmax_idx_extension=fmax_idx_extension,
        )

        ## Define Model
        self.model = BayesianModel(
            self.data,
            degree_fluctuate=degree_fluctuate,
            init_params=init_params,
            Nbw=Nbw,
        )
        self.variation_factor = variation_factor
        

    def run(
        self,
        lr_map: float = 5e-4,
        ntrain_map: int = 5000,
        inference_size: int = 500,
        hmc_burnin_steps: int = 1000,
        hmc_step_size: float = 1e-3,
        hmc_num_leapfrog_steps: int = 3,
        hmc_target_accept_prob: float = 0.75,
        hmc_num_steps_between_results: int = 0,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, BayesianModel, List[tf.Tensor]]:
        logger.debug("Starting Model Inference Training..")
        
        self.run_phase1(lr_map, ntrain_map)
        samp = self.run_phase2(
            inference_size=inference_size,
            hmc_burnin_steps=hmc_burnin_steps,
            hmc_step_size=hmc_step_size,
            hmc_num_leapfrog_steps=hmc_num_leapfrog_steps,
            hmc_target_accept_prob=hmc_target_accept_prob,
            hmc_num_steps_between_results=hmc_num_steps_between_results,
        )

        return self.kdl_losses, self.lp, self.model, samp

    def run_phase1(
        self,
        lr_map: float = 5e-4,
        ntrain_map: int = 5000,
    ):
        """set model.trainable_vars to MAP values

        # Note: at this point, the model.trainable_vars are set to
        # param values that maximise the log posterior
        # THESE ARE NO LONGER CHANGED AFTER PHASE 1
        """
        optimizer_hs = Adam(lr_map)
        start_map = timeit.default_timer()
        logger.debug(f"Start Phase 1: MAP search ({ntrain_map} steps)... ")
        ntrain_map = tf.constant(ntrain_map, dtype=tf.int32)

        @tf.function(reduce_retracing=True)
        def tune_model_to_map(
            model: BayesianModel, optimizer: Adam, n_train: int
        ) -> Tuple[List[tf.Variable], tf.Tensor]:
            n_samp = model.trainable_vars[0].shape[0]
            lpost = tf.constant(0.0, tf.float32, [n_samp])
            lp = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            for i in tf.range(n_train):
                lpost = model.map_train_step(optimizer)

                if optimizer.iterations % 5000 == 0:
                    tf.print(
                        "Step",
                        optimizer.iterations,
                        "/",
                        n_train,
                        " : log posterior",
                        lpost,
                    )
                lp = lp.write(tf.cast(i, tf.int32), lpost)

            return model.trainable_vars, lp.stack()

        opt_vars_hs, self.lp = tune_model_to_map(
            self.model, optimizer_hs, ntrain_map
        )
        self.map_time = timeit.default_timer() - start_map
        logger.debug(f"MAP Training Time: {self.map_time:.2f}s")


    def run_phase2(
        self,
        inference_size: int = 500,
        hmc_burnin_steps: int = 1000,
        hmc_step_size: float = 1e-3,
        hmc_num_leapfrog_steps: int = 3,
        hmc_target_accept_prob: float = 0.75,
        hmc_num_steps_between_results: int = 0,
        **kwargs,
    ):
        """
        Phase 2 UQ with single-chain HMC initialized from the Phase 1 MAP.
        """
        
        initial_state = [
            tf.identity(param[0]) for param in self.model.trainable_vars
        ]

        def conditioned_log_prob(*z):
            params = [tf.expand_dims(zi, axis=0) for zi in z]
            log_prob = self.model.loglik(params) + self.model.logprior(params)
            return tf.squeeze(log_prob, axis=0)

        print("Start Phase 2: HMC sampling...")
        start_hmc = timeit.default_timer()

        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=conditioned_log_prob,
            step_size=tf.constant(hmc_step_size, dtype=tf.float32),
            num_leapfrog_steps=hmc_num_leapfrog_steps,
        )
        adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel,
            num_adaptation_steps=int(0.8 * hmc_burnin_steps),
            target_accept_prob=hmc_target_accept_prob,
        )

        @tf.function(reduce_retracing=True)
        def sample_from_hmc(current_state):
            return tfp.mcmc.sample_chain(
                num_results=inference_size,
                current_state=current_state,
                kernel=adaptive_kernel,
                num_burnin_steps=hmc_burnin_steps,
                num_steps_between_results=hmc_num_steps_between_results,
                trace_fn=lambda _, pkr: (
                    pkr.inner_results.is_accepted,
                    pkr.inner_results.accepted_results.target_log_prob,
                ),
            )

        self.hmc_samples, trace = sample_from_hmc(initial_state)
        self.hmc_is_accepted, self.hmc_target_log_prob = trace
        self.hmc_acceptance_rate = tf.reduce_mean(
            tf.cast(self.hmc_is_accepted, tf.float32)
        )
        self.kdl_losses = -tf.reshape(self.hmc_target_log_prob, [-1])
        self.hmc_time = timeit.default_timer() - start_hmc
        
        print(f"HMC Time: {self.hmc_time:.2f}s")
        print(
            f"HMC acceptance rate: {self.hmc_acceptance_rate.numpy():.3f}"
        )
        self.total_time = self.map_time + self.hmc_time
        print(f"Total Inference Training Time: {self.total_time:.2f}s")

        self.posteriorPointEst = [
            tf.reduce_mean(sample, axis=0) for sample in self.hmc_samples
        ]
        self.posteriorPointEstStd = [
            tf.math.reduce_std(sample, axis=0) for sample in self.hmc_samples
        ]
        self.variationalDistribution = None
        
        return self.hmc_samples

    
