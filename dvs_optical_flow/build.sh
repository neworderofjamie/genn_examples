#!/bin/bash
CXXFLAGS="$(pkg-config --cflags opencv4)" LDFLAGS="$(pkg-config --libs opencv4) -lcaer" genn-buildmodel.sh model.cc
