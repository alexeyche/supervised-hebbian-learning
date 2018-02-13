#!/usr/bin/env bash

wget -O "eigen.tar.gz" "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"

mkdir -p eigen
tar -xf eigen.tar.gz -C eigen

mv eigen/eigen-*/* eigen
