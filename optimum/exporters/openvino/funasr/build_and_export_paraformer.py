import logging
import os
import json
import random
import numpy as np
import torch
import string
import copy
from omegaconf import OmegaConf, DictConfig, ListConfig
from optimum.exporters.openvino.funasr.modeling_paraformer import Paraformer

def add_file_root_path(model_or_path: str, file_path_metas: dict, cfg={}):

    if isinstance(file_path_metas, dict):
        if isinstance(cfg, list):
            cfg.append({})

        for k, v in file_path_metas.items():
            if isinstance(v, str):
                p = os.path.join(model_or_path, v)
                if os.path.exists(p):
                    if isinstance(cfg, dict):
                        cfg[k] = p
                    elif isinstance(cfg, list):
                        # if len(cfg) == 0:
                        # cfg.append({})
                        cfg[-1][k] = p

            elif isinstance(v, dict):
                if isinstance(cfg, dict):
                    if k not in cfg:
                        cfg[k] = {}
                    add_file_root_path(model_or_path, v, cfg[k])
                # elif isinstance(cfg, list):
                #     cfg.append({})
                #     add_file_root_path(model_or_path, v, cfg)
            elif isinstance(v, (list, tuple)):
                for i, vv in enumerate(v):
                    if k not in cfg:
                        cfg[k] = []
                    if isinstance(vv, str):
                        p = os.path.join(model_or_path, vv)
                        # file_path_metas[i] = p
                        if os.path.exists(p):
                            if isinstance(cfg[k], dict):
                                cfg[k] = p
                            elif isinstance(cfg[k], list):
                                cfg[k].append(p)
                    elif isinstance(vv, dict):
                        add_file_root_path(model_or_path, vv, cfg[k])

    return cfg

def get_or_download_model_dir_hf(
    model,
    model_revision=None,
    is_training=False,
    check_latest=True,
):
    """Get local model directory or download model if necessary.

    Args:
        model (str): model id or path to local model directory.
        model_revision  (str, optional): model version number.
        :param is_training:
    """
    from huggingface_hub import snapshot_download

    model_cache_dir = snapshot_download(model)
    return model_cache_dir

def download_from_hf(**kwargs):
	model_or_path = kwargs.get("model")
	model_revision = kwargs.get("model_revision", "master")
	if not os.path.exists(model_or_path) and "model_path" not in kwargs:
		try:
			model_or_path = get_or_download_model_dir_hf(
				model_or_path,
				model_revision,
				is_training=kwargs.get("is_training"),
				check_latest=kwargs.get("check_latest", True),
				)
		except Exception as e:
			print(f"Download: {model_or_path} failed!: {e}")

	kwargs["model_path"] = model_or_path if "model_path" not in kwargs else kwargs["model_path"]

	if os.path.exists(os.path.join(model_or_path, "configuration.json")):
		with open(os.path.join(model_or_path, "configuration.json"), "r", encoding="utf-8") as f:
			conf_json = json.load(f)
			cfg = {}
			if "file_path_metas" in conf_json:
				add_file_root_path(model_or_path, conf_json["file_path_metas"], cfg)
			cfg.update(kwargs)
			if "config" in cfg:
				config = OmegaConf.load(cfg["config"])
				kwargs = OmegaConf.merge(config, cfg)
				kwargs["model"] = config["model"]
	elif os.path.exists(os.path.join(model_or_path, "config.yaml")):
		config = OmegaConf.load(os.path.join(model_or_path, "config.yaml"))
		kwargs = OmegaConf.merge(config, kwargs)
		init_param = os.path.join(model_or_path, "model.pt")
		if "init_param" not in kwargs or not os.path.exists(kwargs["init_param"]):
			kwargs["init_param"] = init_param
			assert os.path.exists(kwargs["init_param"]), "init_param does not exist"
		if os.path.exists(os.path.join(model_or_path, "tokens.json")):
			kwargs["tokenizer_conf"]["token_list"] = os.path.join(model_or_path, "tokens.json")
		if os.path.exists(os.path.join(model_or_path, "seg_dict")):
			kwargs["tokenizer_conf"]["seg_dict"] = os.path.join(model_or_path, "seg_dict")
		kwargs["model"] = config["model"]
		if os.path.exists(os.path.join(model_or_path, "am.mvn")):
			kwargs["frontend_conf"]["cmvn_file"] = os.path.join(model_or_path, "am.mvn")
	if isinstance(kwargs, DictConfig):
		kwargs = OmegaConf.to_container(kwargs, resolve=True)

	return kwargs

def deep_update(original, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            if len(value) == 0:
                original[key] = value
            deep_update(original[key], value)
        else:
            original[key] = value

def load_pretrained_model(
    path: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool = True,
    map_location: str = "cpu",
    oss_bucket=None,
    scope_map=[],
    excludes=None,
    **kwargs,
):
    """Load a model state and set it to the model.

    Args:
            init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:

    """

    obj = model
    dst_state = obj.state_dict()
    ori_state = torch.load(path, map_location=map_location)

    src_state = copy.deepcopy(ori_state)
    src_state = src_state["state_dict"] if "state_dict" in src_state else src_state
    src_state = src_state["model_state_dict"] if "model_state_dict" in src_state else src_state
    src_state = src_state["model"] if "model" in src_state else src_state

    if isinstance(scope_map, str):
        scope_map = scope_map.split(",")
    scope_map += ["module.", "None"]
    logging.info(f"scope_map: {scope_map}")

    for k in dst_state.keys():
        excludes_flag = False
        if excludes is not None:
            for k_ex in excludes:
                if k.startswith(k_ex):
                    logging.info(f"key: {k} matching: {k_ex}, excluded")
                    excludes_flag = True
                    break
        if excludes_flag:
            continue

        k_src = k

        if scope_map is not None:
            src_prefix = ""
            dst_prefix = ""
            for i in range(0, len(scope_map), 2):
                src_prefix = scope_map[i] if scope_map[i].lower() != "none" else ""
                dst_prefix = scope_map[i + 1] if scope_map[i + 1].lower() != "none" else ""

                if dst_prefix == "" and (src_prefix + k) in src_state.keys():
                    k_src = src_prefix + k
                    if not k_src.startswith("module."):
                        logging.info(f"init param, map: {k} from {k_src} in ckpt")
                elif (
                    k.startswith(dst_prefix)
                    and k.replace(dst_prefix, src_prefix, 1) in src_state.keys()
                ):
                    k_src = k.replace(dst_prefix, src_prefix, 1)
                    if not k_src.startswith("module."):
                        logging.info(f"init param, map: {k} from {k_src} in ckpt")

        if k_src in src_state.keys():
            if ignore_init_mismatch and dst_state[k].shape != src_state[k_src].shape:
                logging.info(
                    f"ignore_init_mismatch:{ignore_init_mismatch}, dst: {k, dst_state[k].shape}, src: {k_src, src_state[k_src].shape}"
                )
            else:
                dst_state[k] = src_state[k_src]
        else:
            print(f"Warning, miss key in ckpt: {k}, {path}")

    flag = obj.load_state_dict(dst_state, strict=True)

def _torchscripts(model, path, device="cuda"):
    dummy_input = model.export_dummy_inputs()
    model_jit_script = torch.jit.trace(model, dummy_input)
    return model_jit_script

def export_utils(
    model, data_in=None, quantize: bool = False, opset_version: int = 14, type="onnx", **kwargs
):
    model_scripts = model.export(**kwargs)
    export_dir = kwargs.get("output_dir", os.path.dirname(kwargs.get("init_param")))
    os.makedirs(export_dir, exist_ok=True)

    if not isinstance(model_scripts, (list, tuple)):
        model_scripts = (model_scripts,)
    for m in model_scripts:
        m.eval()
        device = "cpu"    
        print("Exporting torchscripts on device {}".format(device))
        model_jit_scripts = _torchscripts(m, path=export_dir, device=device)

    return export_dir, model_jit_scripts

########################################
####### API for main program
########################################
def download_model(**kwargs):
	kwargs = download_from_hf(**kwargs)
	return kwargs

def build_model(**kwargs):
	assert "model" in kwargs
	kwargs = download_model(**kwargs)
	torch.set_num_threads(kwargs.get("ncpu", 4))

	# build tokenizer
	# Here to remove building tokenizer to get vocab_size. Currently hard_code the value here
	# Check the downloaded token.json and the vocab_size is the token number in token.json
	kwargs["vocab_size"] = 8404

	# build model
	model_conf = {}
	deep_update(model_conf, kwargs.get("model_conf", {}))
	deep_update(model_conf, kwargs)
	model = Paraformer(**model_conf)

	# init_param
	init_param = kwargs.get("init_param", None)
	if init_param is not None:
		if os.path.exists(init_param):
			logging.info(f"Loading pretrained params from {init_param}")
			load_pretrained_model(
				model=model,
				path=init_param,
				ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
				oss_bucket=kwargs.get("oss_bucket", None),
				scope_map=kwargs.get("scope_map", []),
				excludes=kwargs.get("excludes", None),
			)
		else:
			print(f"error, init_param does not exist!: {init_param}")

	# fp16
	if kwargs.get("fp16", False):
		model.to(torch.float16)
	elif kwargs.get("bf16", False):
		model.to(torch.bfloat16)
	model.to(kwargs["device"])

	return model, kwargs

def export(model, kwargs, input=None, **cfg):
	del kwargs["model"]
	model.eval()

	with torch.no_grad():
		export_dir, model_jit_scripts = export_utils(model=model, **kwargs)

	return export_dir, model_jit_scripts
