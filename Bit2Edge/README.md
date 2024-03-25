# Bit2EdgeV2-BDE Repository

# Module

- config: Holding the current global state of the current process

    + devConfig: Storing the state of learning_rate and train/dev/test for model's training

    + userConfig: Storing the state of the fingerprint input, the model's setup used in training (later migration), the
      random_seed (later migration), the computing device, the visualization. Don't modify this class during runtime and
      let it go unmanaged.

- dataObject: The class of storing the data

    + Dataset.py: The main class of storing the data.

    + DatasetLinker.py: The parent class of any functioning classes that connected to the class::FeatureData in
      Dataset.py

    + DatasetUtils.py: Some utilities function adapted for the Dataset.py

    + FileParseParams.py: The small class to allow the passing when user input the reading file with this setup (I/O)

    + InputState.py: The __static__ class holds to store the current state of the input setup in userConfig.py

- input: Manage the feature generator/pipelining of the model

    + Fingerprint: A module list of utilized objects to generate the hashed bit-type molecular fingerprints.

    + LBI_Feat: A module list of utilized objects to generate the localized bond information.

    + AtomBondTips.py: Some utility functions to overcome the bottleneck of RDKit molecule processing.

    + BVCreator.py: The object that grouped ONE bond/radical environment by many FingerprintGenerator.

    + BVManager.py: The object that grouped many class::BVCreator.

    + EnvExtractor.py: The object that split the molecule into bond/radical environment(s) whose setting is captured in
      the InputState.py and userConfig.py

    + FeatureUtils.py: The utility functions to called from Dataset.py to analyze the current label of features.

    + LBondInfo.py: The object that group and call the feature creation that have been setup in here and executed by the
      feature generator in module LBI_Feat.

    + MolEngine.py: Additional class to perform molecule pre-processing.

    + MolUtility.py: Controlled class to perform quicker bond routing and traversing.

    + SubgraphUtils.py: The core algorithm of bond/radical environment extractor for class::EnvExtractor.

- model: Manage the core model to deploy and control how input is fed into the TensorFlow model

    + data.py: Control how input is fed into the TensorFlow model

    + layer.py: Some utility functions to seamlessly deploy the layer of the model

    + model.py (B2E_Model): The class to build, deploy your model and doing prediction with custom data visualization (
      for testing)

    + utils.py: Some legacy utility functions to manage.

- test: The module to initiate the Tester for prediction.

    + BaseTester.py: The parent class of the Tester, controlling the input of the user.

    + EnsModel.py: The sub-model designed to predict your BDE from multiple models using Linear Regression algorithm

    + MultiPredictor.py: The exposed class to predict your BDE using class::EnsModel.

    + Tester.py: The exposed class to predict your BDE using class::B2E_Model.

    + TesterUtils.py: Some supported function of class::Tester.

- train:

    + DatasetSplitter.py: The object to split the dataset into Train/Dev/Test

    + Trainer.py: The exposed class to train your model using class::B2E_Model.

    + TrainerUtils.py: Some supported function of class::Trainer.

- utils:

    + helper.py: Some utility functions that mostly used to control the file reading, sorting, I/O, analyze the dataset,
      ...

    + verify.py: Some utility functions to control the data-type to let the program be static and not let incorrect
      datatype.
