#!/bin/bash

# Tarik pembaruan terbaru dari GitHub
git pull origin main

# Gunakan 'python' yang aktif saat ini, jangan gunakan jalur absolut absolut /opt/conda/...
python -m pip install releases/*.whl --force-reinstall
