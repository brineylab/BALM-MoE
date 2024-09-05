#!/usr/bin/python
# filename: base_config.py

#
# Copyright (c) 2024 Bryan Briney
# License: GNU General Public License, version 3.0 (http://opensource.org/licenses/gpl-3-0/)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import json
import os
from typing import Optional


class BaseConfig:
    def to_json(self, output: Optional[str] = None):
        """
        Save the config to a JSON file.

        Parameters
        ----------
        output : str, optional
            The path to the JSON file to save the config to.
            If None, the config is returned as a JSON string.
        """
        json_string = json.dumps(self.__dict__)
        if output is not None:
            with open(output, "w") as f:
                f.write(json_string)
        else:
            return json_string

    def save_pretrained(self, save_directory: str):
        """
        Save the config to a JSON file.

        Parameters
        ----------
        save_directory : str
            The directory to save the config to.
        """
        self.to_json(os.path.join(save_directory, "config.json"))

    @classmethod
    def from_json(cls, json_path: str):
        """
        Load the config from a JSON file.

        Parameters
        ----------
        json_path : str
            The path to the JSON file containing the config.
        """
        with open(json_path, "r") as f:
            json_string = f.read()
        return cls(**json.loads(json_string))

    @classmethod
    def from_dict(cls, config: dict):
        """
        Load the config from a dictionary.

        Parameters
        ----------
        config : dict
            The config as a dictionary.
        """
        return cls(**config)

    @classmethod
    def from_pretrained(cls, pretrained_config_or_path: str):
        """
        Load the config from a pretrained model.

        Parameters
        ----------
        pretrained_model_path : Union[str, dict, BaseConfig]
            The pretrained model config. Can be any of the following:
                - A string path to a JSON file containing the config.
                - A dictionary containing the config.
                - A BaseConfig object.
        """
        if isinstance(pretrained_config_or_path, str):
            pretrained_config_or_path = os.path.abspath(pretrained_config_or_path)
            # input is the config file
            if os.path.isfile(pretrained_config_or_path):
                return cls.from_json(pretrained_config_or_path)
            # input is a directory containing the config file
            elif os.path.isdir(pretrained_config_or_path) and os.path.exists(
                os.path.join(pretrained_config_or_path, "config.json")
            ):
                return cls.from_json(
                    os.path.join(pretrained_config_or_path, "config.json")
                )
            else:
                raise ValueError(
                    f"Invalid pretrained model path: {pretrained_config_or_path}"
                )

        elif isinstance(pretrained_config_or_path, dict):
            # input is a the config as a dictionary
            return cls.from_dict(pretrained_config_or_path)
        elif isinstance(pretrained_config_or_path, BaseConfig):
            # input is a BaseConfig object
            return pretrained_config_or_path
        else:
            raise ValueError(
                f"Invalid input: {pretrained_config_or_path}. Expected a string, dict, or BaseConfig."
            )
