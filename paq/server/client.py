#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import requests
import time

task = {"query": "who sang does he love me with reba", "k": 10}
resp = requests.post("http://0.0.0.0:1359/", json=task)
data = resp.json()
print(data)
