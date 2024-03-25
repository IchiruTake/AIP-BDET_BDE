# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Placeholder Design
# BasePlaceholder --> SinglePlaceholderV1 --> SinglePlaceholderV2


from typing import Dict, Optional

from Bit2Edge.config.ConfigUtils import ReadYamlConfigFile
from Bit2Edge.utils.file_io import FixPath
from Bit2Edge.utils.verify import InputFullCheck, TestState

_CONFIG_KEYS = ['EnvFilePath', 'LBIFilePath', 'SavedInputConfig', 'SavedModelConfig', 'ModelKey']


def LoadConfiguration(EnvFilePath: str, LBIFilePath: str, SavedInputConfigFile: str, SavedModelConfigFile: str,
                      TF_Model: Optional[str] = None) -> Dict[str, str]:
    TestState(EnvFilePath is not None, '[ERROR] Saved EnvFilePath is NOT DEFINED.')
    TestState(LBIFilePath is not None, '[ERROR] Saved LBIFilePath is NOT DEFINED.')
    TestState(SavedInputConfigFile is not None, '[ERROR] Saved input configuration is NOT DEFINED.')
    TestState(SavedModelConfigFile is not None, '[ERROR] Saved model configuration is NOT DEFINED.')
    InputFullCheck(EnvFilePath, name='EnvFilePath', dtype='str', fastCheck=True)
    InputFullCheck(LBIFilePath, name='LBIFilePath', dtype='str', fastCheck=True)
    InputFullCheck(SavedInputConfigFile, name='SavedInputConfigFile', dtype='str', fastCheck=True)
    InputFullCheck(SavedModelConfigFile, name='SavedModelConfigFile', dtype='str', fastCheck=True)

    SavedModelConfig = ReadYamlConfigFile(FilePath=SavedModelConfigFile)
    TestState(SavedModelConfig.get('ModelKey', None) is not None, '[ERROR] No ModelKey is found.')

    result = {
        'EnvFilePath': EnvFilePath,
        'LBIFilePath': LBIFilePath,
        'SavedInputConfig': SavedInputConfigFile,
        'SavedModelConfig': SavedModelConfigFile,
        'ModelKey': SavedModelConfig.get('ModelKey'),
    }
    if TF_Model is not None:
        result['TF_Model'] = FixPath(TF_Model, extension='.h5')
    return result
