import sys
from src.common.constants import PROJECT_FECNET


def syspath_append_projects():
    sys.path.extend([
        PROJECT_FECNET,
    ])
