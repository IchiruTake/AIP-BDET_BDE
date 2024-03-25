# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

import yaml

from Bit2Edge.utils.file_io import FixPath


def WriteDictToYamlFile(FilePath: str, DataConfiguration: dict) -> None:
    if not isinstance(FilePath, str):
        raise TypeError(':arg:`FilePath` must be a string for data storage.')
    f = open(fr'{FixPath(FileName=FilePath, extension=".yaml")}', 'w')
    return yaml.dump(data=DataConfiguration, stream=f)


def ReadYamlConfigFile(FilePath: str) -> dict:
    f = open(fr'{FixPath(FileName=FilePath, extension=".yaml")}', 'r')
    return yaml.load(stream=f, Loader=yaml.FullLoader)
