
# Safe Learning on Real Robots
A collection of algorithms and experiment tools for safe sim to real transfer and learning in robotics.

## Features
* Three different CMDP solvers, [CRPO](https://arxiv.org/abs/2011.05869), [Saute-RL](https://arxiv.org/abs/2202.06558) and primal-dual, compatible with (variants of) [Brax's](https://github.com/google/brax) SAC, MBPO and PPO.
* Algorithm implementation is interchangeable between training in simulation to training on real robots via [OnlineEpisodeOrchestrator](https://github.com/yardenas/safe-sim2real/blob/b1962ac5e453b172eaa8373f8d382b562cf5a03b/ss2r/rl/online.py#L19C7-L19C32). Check out `rccar_experiments` for a full example. Support for training online on any real robot supported by [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), including Unitree Go1/2.
* Blazing fast training. Full compatibility with [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground). Reimplementation of OpenAI's [Safety Gym](https://github.com/openai/safety-gym) in MJX and safety tasks from [Real-World RL suite](https://github.com/google-research/realworldrl_suite/tree/master).


## Requirements

- Python == 3.11.6
- [`uv`](https://docs.astral.sh/uv/) (recommended) or the built-in `venv`

## Installation

### Using pip

```bash
git clone https://github.com/yardenas/safe-learning
cd safe-learning
python3 -m venv venv
source venv/bin/activate
pip install -e .
````

### Using uv

Install uv if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a project environment and install dependencies:

```bash
git clone https://github.com/yardenas/safe-learning
cd safe-learning
uv sync
uv run python --version  # sanity check, optional
```

### Installing `madrona_mjx` (optional, required for Madrona backend)

Some benchmarks (e.g., the MJX-based pick-and-place tasks) require the custom
[`madrona_mjx`](https://github.com/shacklettbp/madrona_mjx) fork. Build and
install it inside the UV environment you created above:

1. From the parent directory of `safe-sim2real`, clone the repository and check
   out the tested commit:

   ```bash
   git clone https://github.com/shacklettbp/madrona_mjx.git
   cd madrona_mjx
   git checkout c34f3cf6d95148dba50ffeb981aea033b8a4d225
   git submodule update --init --recursive
   ```

2. Configure and build (disable Vulkan if you do not have it available):

   ```bash
   mkdir -p build
   cd build
   cmake -DLOAD_VULKAN=OFF ..
   cmake --build . -j
   cd ..
   ```

3. Install the Python bindings into your UV environment:

   ```bash
   uv pip install -e .
   ```

Refer to the upstream repository for platform-specific prerequisites (CUDA,
Vulkan, compiler versions). Re-run `uv pip install -e .` whenever you rebuild the
library.

## Usage

Our code uses [Hydra](https://hydra.cc/) to configure experiments. Each experiment is defined as a `yaml` file in `ss2r/configs/experiments`. For example, to train a Unitree Go1 policy with a constraint on joint limit:

```bash
python train_brax.py +experiment=go1_sim_to_real
```
## Docs
* Policies (in `onnx` format) used for the Unitree Go1 experiments can be found in `ss2r/docs/policies`.
* In `ss2r/docs/videos` you can find videos of 5 trials for each policy, marked by its policy id.


## Citation

If you find our repository useful in your work, please consider citing:

```bibtex
@inproceedings{
  as2025spidrsimpleapproachzeroshot,
  title={{SP}i{DR}: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer},
  author={Yarden As and Chengrui Qu and Benjamin Unger and Dongho Kang and Max van der Hart and Laixi Shi and Stelian Coros and Adam Wierman and Andreas Krause},
  booktitle={International Conference on Neural Information Processing Systems},
  year={2025},
}
```

<!-- ## Learn More

* **Project Webpage**: 
* **Paper**:
* **Contact**: 
