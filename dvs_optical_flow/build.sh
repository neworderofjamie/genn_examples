#!/bin/bash
CXXFLAGS="$(pkg-config --cflags opencv4)" LDFLAGS="$(pkg-config --libs opencv4)" genn-buildmodel.sh -e model.cc
