"""
RapidVision - AI-powered object detection and distance estimation package.
"""

__version__ = "0.0.0"

from . import camera
from . import detection
from . import utils
from . import config

'''
RapidVision/
│
├── 📁 rapidvision/               # Main package
│   ├── __init__.py
│   ├── main.py                   # Entry point (can import from below modules)
│   ├── camera/                   # All cam calibration logic
│   │   └── calibration.py
│   ├── detection/                # Unified detection API
│   │   ├── yolox_wrapper.py
│   │   ├── ppyoloe_wrapper.py
│   │   └── shared_data.py
│   ├── utils/                    # Common utilities
│   │   └── general.py
│   └── config/                   # JSONs, configs
│       ├── cam_cali_data.json
│       └── object_dimensions.json
│
├── 📁 third_party/               # Vendor folders: untouched model codebases
│   ├── yolox/
│   └── ppyoloe/
│
├── 📁 models/                    # Just pretrained model files
│   ├── ppyoloe_ready.pt
│
├── README.md
└── LICENSE.md
'''

# run main.py with: PYTHONPATH=third_party python main.py