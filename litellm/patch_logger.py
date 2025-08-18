import logging
import sys

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)

patch_logger = logging.getLogger("LiteLLM Patch")
patch_logger.addHandler(handler)
patch_logger.setLevel(logging.INFO)
