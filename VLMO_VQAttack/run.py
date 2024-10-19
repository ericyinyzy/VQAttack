import os
import copy
import pytorch_lightning as pl

from vlmo.config import ex
from vlmo.modules import VLMo
from vlmo.datamodules.multitask_datamodule import MTDataModule

from pytorch_lightning.plugins import environments as pl_env
from pytorch_lightning.utilities.distributed import rank_zero_info
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#wo/ w
class OMPIClusterEnvironment(pl_env.ClusterEnvironment):
    def __init__(self):
        super().__init__()

    # def creates_children(self) -> bool:
    #     # return True if the cluster is managed (you don't launch processes yourself)
    #     assert (
    #         "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ
    #     )  # this cluster is managed
    #     return True

    @property
    def creates_processes_externally(self):
        return True

    def world_size(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])

    def set_world_size(self, size: int):
        pass

    def global_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])

    def set_global_rank(self, rank: int):
        pass

    def local_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    def node_rank(self) -> int:
        if "NODE_RANK" in os.environ:
            return int(os.environ["NODE_RANK"])
        else:
            return 0

    def master_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    def master_port(self) -> int:
        return int(os.environ["MASTER_PORT"])


def get_cluster_plugin(num_gpus=1, num_nodes=1):
    if num_nodes > 1 or (
        num_nodes == 1 and "OMPI_COMM_WORLD_SIZE" in os.environ
    ):
        rank_zero_info("ClusterPlugin: using OMPI Cluster Environment")
        return OMPIClusterEnvironment()
    if num_gpus >= 1:
        rank_zero_info("ClusterPlugin: using Lightning Cluster Environment")
        return pl_env.LightningEnvironment()
    return None


@ex.automain
def main(_config):
    # print(_config,type(_config))
    # exit()
    # _config={'exp_name': 'mlm_itm_itc_base', 'seed': 1, 'datasets': ['vqa'], 'loss_names': {'itm': 1, 'itc': 1, 'mlm': 1, 'textmlm': 0, 'vqa': 0, 'nlvr2': 0, 'irtr': 0}, 'batch_size': 1024, 'train_transform_keys': ['square_transform_randaug'], 'val_transform_keys': ['square_transform'], 'image_size': 224, 'draw_false_image': 0, 'image_only': False, 'text_only': False, 'vqav2_label_size': 3129, 'max_text_len': 40, 'max_text_len_of_initckpt': 196, 'tokenizer': 'bert-base-uncased', 'vocab_size': 30522, 'whole_word_masking': True, 'mlm_prob': 0.15, 'draw_false_text': 0, 'model_arch': 'vlmo_base_patch16', 'drop_path_rate': 0.1, 'optim_type': 'adamw', 'learning_rate': 0.0002, 'weight_decay': 0.01, 'decay_power': 1, 'max_epoch': 100, 'max_steps': 200000, 'warmup_steps': 0.1, 'end_lr': 0, 'lr_mult': 1, 'get_recall_metric': False, 'get_recall_rerank_metric': False, 'k_test': 32, 'resume_from': None, 'fast_dev_run': False, 'val_check_interval': 1.0, 'test_only': True, 'use_sharded_training': False, 'resume_during_training': False, 'data_root': '/data/ziyi/vlmo_data/VQA_arrow', 'log_dir': 'result', 'per_gpu_batchsize': 1, 'num_gpus': 1, 'num_nodes': 1, 'load_path': '/data/ziyi/vlmo_model/vlmo_base_patch16_224.pt', 'num_workers': 8, 'precision': 32}
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    # print('VLMo', VLMo)
    # exit()
    model = VLMo(_config)

    # exit()
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    rank_zero_info("grad_steps: {}".format(grad_steps))

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    resume_ckpt = None
    if _config["resume_during_training"]:
        for index in range(100):
            ckpt_path = os.path.join(_config["log_dir"], f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}', "version_{}/checkpoints/last.ckpt".format(index))
            if os.path.exists(ckpt_path):
                resume_ckpt = ckpt_path
    
    rank_zero_info("resume_ckpt: {}".format(resume_ckpt))

    cluster_plugin = get_cluster_plugin(
        _config["num_gpus"], _config["num_nodes"]
    )
    plugin_list = [cluster_plugin]
    rank_zero_info("plugin_list: {}".format(plugin_list))

    if _config["use_sharded_training"]:
        rank_zero_info("Using ddp sharded")
        distributed_strategy = "ddp_sharded"
    else:
        distributed_strategy = "ddp"
    # print('xxxxxx')
    # exit()
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="gpu",
        strategy=distributed_strategy,
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=resume_ckpt,
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        plugins=plugin_list,
    )

    if _config["loss_names"]["textmlm"] > 0:
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            for key in ["text_embeddings", "token_type_embeddings", "mlp_text", "norm2_text", "mlm_score", "relative_position_bias_table", "transformer.norm"]:
                if key in name:
                    param.requires_grad = True

        for name, param in model.named_parameters():
            rank_zero_info("{}\t{}".format(name, param.requires_grad))
    # print('xxxxxx')
    # exit()
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        # print(trainer.test)
        # print(dm.tokenizer)
        # exit()
        print('aaaaaa')
        # exit()
        # exit()
        trainer.test(model, datamodule=dm)
