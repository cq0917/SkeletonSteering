import optax

from loco_mujoco.algorithms import AMPJax as _AMPJax


class AMPJax(_AMPJax):
    @classmethod
    def _get_optimizer(cls, config):
        max_grad_norm = config.experiment.max_grad_norm
        disc_max_grad_norm = getattr(config.experiment, "disc_max_grad_norm", max_grad_norm)

        if config.experiment.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adamw(weight_decay=config.experiment.weight_decay, eps=1e-5,
                            learning_rate=lambda count: cls._linear_lr_schedule(
                                count,
                                config.experiment.num_minibatches,
                                config.experiment.update_epochs,
                                config.lr,
                                config.experiment.num_updates,
                            )),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adamw(config.experiment.lr, weight_decay=config.experiment.weight_decay, eps=1e-5),
            )

        disc_tx = optax.chain(
            optax.clip_by_global_norm(disc_max_grad_norm),
            optax.adamw(config.experiment.disc_lr, weight_decay=config.experiment.weight_decay, eps=1e-5),
        )

        return tx, disc_tx
