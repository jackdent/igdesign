from hydra.utils import instantiate
import pytorch_lightning as pl

from igdesign.features.feature_factory import init_feature_factory


class Model(pl.LightningModule):
    def __init__(self, cfg, *ignore_args, **kwargs):
        super().__init__()
        try:
            self.save_hyperparameters()
        except:
            print("[WARNING] Model could not save model hyperparameters")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.config = cfg
        self.overrides = kwargs

        # --------------------------------------------------------------------------
        # We don't want to seed everything when we load the model, since this leads to
        # unintuiive behaviour (loading the model changes state).
        # --------------------------------------------------------------------------
        # seed = self.configure("random_seed")
        # if seed is not None:
        #     pl.seed_everything(seed)

        self.task = self.configure("task")
        feature_factory_kwargs = (
            self.task["feature_factory_kwargs"] if self.task else {}
        )

        feature_config = self.configure("features")
        self.feature_factory = init_feature_factory(
            feature_config, **feature_factory_kwargs
        )

        model_config = self.configure("model")
        if model_config is not None:
            self.init_model(model_config)

    def configure(self, name, config=None):
        if config is None:
            config = self.config
        if name in self.overrides:
            return self.overrides[name]
        elif name in config:
            return config[name]
        else:
            return None

    def init_model(self, model_cfg):
        self.model = instantiate(model_cfg)

    def forward(self, batch, mode="train"):
        """Single forward pass through the model."""
        return self.model.forward(
            batch,
        )
