#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
"""
Groups together code used for creating a NuPIC model and dealing with IO.
(This is a component of the One Hot Gym Anomaly Tutorial.)
"""
import importlib
import sys
import csv
import datetime
import dateutil

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory

import nupic_anomaly_output


MODEL_PARAMS_DIR = "./model_params"

def createModel(modelParams):
  model = ModelFactory.create(modelParams)
  model.enableInference({"predictedField": "ECG1"})
  return model



def getModelParams():
  importName = "model_params.model_params"
  print "Importing model params from %s" % importName
  importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  return importedModelParams



def runIoThroughNupic(inputData, model, gymName, plot):
  inputFile = open(inputData, "rb")
  csvReader = csv.reader(inputFile)
  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  shifter = InferenceShifter()
  if plot:
    output = nupic_anomaly_output.NuPICPlotOutput(gymName)
  else:
    output = nupic_anomaly_output.NuPICFileOutput(gymName)

  counter = 0
  for row in csvReader:
    counter += 1
    if (counter % 100 == 0):
      print "Read %i lines..." % counter
    timestamp = dateutil.parser.parse(row[0])
    ecg = float(row[1])
    result = model.run({
      "timestamp": timestamp,
      "ECG1": ecg
    })

    if plot:
      result = shifter.shift(result)

    prediction = result.inferences["multiStepBestPredictions"][1]
    anomalyScore = result.inferences["anomalyScore"]
    output.write(timestamp, ecg, prediction, anomalyScore)

  inputFile.close()
  output.close()



def runModel(gymName, plot=False):
  model = createModel(getModelParams())
  inputData = "../src/experiments/ecg1_chfdbchf13/chfdbchf13_final.csv"
  runIoThroughNupic(inputData, model, gymName, plot)



if __name__ == "__main__":
  plot = False
  args = sys.argv[1:]
  if "--plot" in args:
    plot = True
  runModel("ECG1", plot=plot)
