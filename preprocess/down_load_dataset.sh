#!/bin/bash

DATASET_NAME=xuekai/flowrl-data-collection

hf download $DATASET_NAME \
  --repo-type dataset \
  --local-dir data/$DATASET_NAME \