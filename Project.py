'''
Author: your name
Date: 2021-11-06 09:36:32
LastEditTime: 2021-11-06 14:39:55
LastEditors: Please set LastEditors
Description: Project class 初始化用到的超
FilePath: \PetFinderNew\Project.py
'''
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Project:
    """
    This class represents our project. It stores useful information about the structure, e.g. paths.
    """
    base_dir: Path = Path(__file__).parents[0]
    data_dir = base_dir / 'dataset'
    checkpoint_dir = base_dir / 'checkpoint'
    configPath: Path = "D\\Work\\PetFinderNew"

    def __post_init__(self):
        # create the directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)