#!/bin/bash

sed -r -i "s/\(\[(.*)\]\(\1\)\)/\(\1\)/g" $1