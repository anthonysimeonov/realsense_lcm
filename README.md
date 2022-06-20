# realsense_lcm

## Setup
Virtualenv setup
```
cd /path/to/venvs
virtualenv -p `which python3` pyrs
source pyrs/bin/activate
```

Python dependencies
```
cd /path/to/realsense_lcm

# install deps
pip install -r requirements.txt

# setup package
pip install -e .
```

LCM -- see [this link](https://lcm-proj.github.io/build_instructions.html) for LCM build instructions. After building LCM, install into python virtualenv
```
cd /path/to/lcm/lcm-python
python setup.py install
```

Check installation
```
python -c "import lcm; print(lcm.__package__); import realsense_lcm; print(realsense_lcm.__package__)"
```

## Run LCM camera publisher
```
python multi_realsense_publisher_visualizer.py
```

## Run LCM camera subscriber
(with publisher running)
```
python rs_sub.py
```

## Notes
If running inside a docker container, you most likely must be running as `root` user to launch the publisher (not the subscriber).

Change the serial numbers in `config/default_multi_realsense_cfg.py` or create a `.yaml` file in `config/real_cam_cfgs` and point to this `.yaml` file with the `--config` flag when running the publisher to override the defaults.

Follow the pattern of the `.lcm` files in `lcm_types` to create a new message type. If you create a new type, you must re-run `lcm-gen` (see [here](https://lcm-proj.github.io/tut_python.html) for more details)
```
cd /path/to/realsense_lcm/lcm_types
lcm-gen -p *.lcm
```
