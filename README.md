# TAPAS_UTILS

This repo contains a utiliy script that you can use for the PyTorch version of the Tapas algorithm, available in the [HuggingFace Transformers library](https://huggingface.co/transformers/model_doc/tapas.html).

The script allows you to automatically create answer coordinates given a table, question and answer texts.

The script is a copy of the [Interaction utils parser](https://github.com/google-research/tapas/blob/master/tapas/utils/interaction_utils_parser.py) of the original repository, but adapted to work for Pandas dataframes instead of Protocol buffers.

There's also an accompanying notebook, showcasing what you can do with the script.
