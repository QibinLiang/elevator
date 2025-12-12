# SPDX-License-Identifier: MIT
# Authors: 
# Qibin Liang <physechan@gmail.com>
# Ning Sun <sunning1888@gmail.com>
# ShuangShuang Zou <547685355@qq.com>
# Organization: Algorithm theory assignment
# Date: 2025-12-12
# License: MIT

import logging
import sys

try:
    import loguru  # type: ignore

    global_logger = loguru.logger

    def set_logger_level(level):
        global_logger.remove()
        global_logger.add(sys.stderr, level=level)

except ImportError:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.Logger.trace = logging.Logger.debug
    global_logger = logging.getLogger("rl-lift")

    def set_logger_level(level):
        global_logger.setLevel(level)


def get_logger():
    return global_logger
