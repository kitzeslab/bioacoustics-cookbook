from bioacoustics_model_zoo import HawkEars
import torch

from opensoundscape.ml.cnn import register_model_cls


@register_model_cls
class CustomLRHawkEars(HawkEars):
    def __init__(self, classes=None, **kwargs):
        super().__init__(**kwargs)
        if classes is not None:
            self.change_classes(classes)

    @classmethod
    def load(cls, path):
        loaded_content = torch.load(path)

        if isinstance(loaded_content, dict):
            # initialize with random weights is not currently supported, so init as normal
            model = cls(classes=loaded_content["classes"])
            # load up the weights and instantiate from dictionary keys
            # includes preprocessing parameters and settings
            state_dict = loaded_content.pop("weights")

            # load weights from checkpoint
            model.network.load_state_dict(state_dict)
        else:
            model = loaded_content  # entire pickled object, not dictionary

        return model

    def recreate_clf(self):
        # use appropriate output dimensions (`len(self.classes)`) for each sub-model's fc layer
        ensemble_list = [
            self.network.model_0,
            self.network.model_1,
            self.network.model_2,
            self.network.model_3,
            self.network.model_4,
        ]

        self.clf_params = []
        for submodel in ensemble_list:
            # initializes new FC layer with random weights
            submodel.head.fc = torch.nn.Linear(2048, len(self.classes))
            self.clf_params.extend(submodel.head.fc.parameters())
        self.network.to(self.device)

    def change_classes(self, classes):
        """changes output layer sizes to match classes

        initializes fc layers of each sub-model with random weights

        also resets torch metrics to align with number of classes
        """
        self.classes = classes
        self._init_torch_metrics()
        self.recreate_clf()

    def configure_optimizers(
        self,
        reset_optimizer=False,
        restart_scheduler=False,
    ):
        """sets lower learning rate  (1/10th) for feature extractor"""

        learning_rate = self.optimizer_params["kwargs"]["lr"]
        if reset_optimizer:
            self.optimizer = None
        if restart_scheduler:
            self.scheduler = None
            self.lr_scheduler_step = -1

        feature_extractor_params = self.network.parameters()
        clf_param_ids = [id(p) for p in self.clf_params]
        feature_extractor_params = [
            p for p in feature_extractor_params if id(p) not in clf_param_ids
        ]

        # create specific subgroups, with
        optimizer = torch.optim.AdamW(
            [
                {"params": feature_extractor_params, "lr": learning_rate / 10},
                {"params": self.clf_params, "lr": learning_rate},
            ],
            lr=learning_rate / 10,
        )

        if hasattr(self, "optimizer") and self.optimizer is not None:
            # load the state dict of the previously existing optimizer,
            # updating the params references to match current instance of self.network
            try:
                opt_state_dict = self.optimizer.state_dict().copy()
                opt_state_dict["params"] = self.network.parameters()
                optimizer.load_state_dict(opt_state_dict)
            except:
                import warnings

                warnings.warn(
                    "attempt to load state dict of existing self.optimizer failed. "
                    "Optimizer will be initialized from scratch"
                )

        # create learning rate scheduler
        # self.scheduler_params dictionary has "class" key and kwargs for init
        # additionally use self.lr_scheduler_step to initialize the scheduler's "last_epoch"
        # which determines the starting point of the learning rate schedule
        # (-1 restarts the lr schedule from the initial lr)
        args = self.lr_scheduler_params["kwargs"].copy()
        args.update({"last_epoch": self.lr_scheduler_step})
        scheduler = self.lr_scheduler_params["class"](optimizer, **args)

        if self.scheduler is not None:
            # load the state dict of the previously existing scheduler,
            # updating the params references to match current instance of self.network
            try:
                scheduler_state_dict = self.scheduler.state_dict().copy()
                # scheduler_state_dict["params"] = self.network.parameters()
                scheduler.load_state_dict(scheduler_state_dict)
            except:
                warnings.warn(
                    "attempt to load state dict of existing self.scheduler failed. "
                    "Scheduler will be initialized from scratch"
                )

        return {"optimizer": optimizer, "scheduler": scheduler}
