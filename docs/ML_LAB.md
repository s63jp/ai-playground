# ML Lab: Researcher Guide

This document describes the Machine Learning lab infrastructure for training neural controllers and studying learned behaviour in the AI Playground.

> **Note:** Features marked "*per interface contract*" describe the architecture as specified in [ML_INTERFACE.md](ML_INTERFACE.md). Final code may differ; verify against the implementation.

## Overview

The ML lab provides:
- **Data generation** from the real physics engine with full provenance tracking
- **Spec format** for training dataset definitions
- **Training pipeline** with experiment tracking
- **Model registry** for researcher and agent neural networks
- **CLI commands** (`playground ml ...`) for common workflows
- **HTTP endpoints** for programmatic access
- **Dashboard integration** for visualization
- **Neural controller adapter** for running trained models as agents

## Data Generation

### Provenance Tracking

Every training example includes:
- `session_id`: Source session from the database
- `action_idx`: Index of the action that produced this measurement
- `state_hash`: Physics state hash at the moment of capture
- `config_hash`: Configuration hash ensuring reproducibility

*Per interface contract*, this allows researchers to trace any prediction back to the exact physics state that produced it.

### Generating Datasets

```bash
# Generate a throw prediction dataset from exploration sessions
playground ml generate --spec specs/throw_apex.yaml --output data/throw_apex/

# Generate from specific sessions
playground ml generate --spec specs/throw_apex.yaml --sessions 1,2,3 --output data/custom/
```

### Spec Format

Dataset specs are YAML files defining:

```yaml
# specs/throw_apex.yaml
name: throw_apex_predictor
version: "1.0"
description: "Predict throw apex from release parameters"

# Source data
source:
  tool: throw
  require_success: true

# Input features (from action args + world state)
features:
  - object_mass
  - release_speed
  - release_direction
  - agent_velocity_at_release
  - held_height

# Target (from action result)
target: max_height_above_launch

# Filtering
filters:
  - outcome: [rest, lost]  # Only completed throws
```

## Training

### Running Training

```bash
# Train a model from a spec
playground ml train --spec specs/throw_apex.yaml --data data/throw_apex/ \
    --epochs 100 --device cpu

# With experiment tracking
playground ml train --spec specs/throw_apex.yaml --data data/throw_apex/ \
    --experiment throw_apex_v2 --tags baseline
```

*Per interface contract*, training uses PyTorch with:
- Automatic feature normalization
- Configurable architecture (MLP default)
- Device selection (CPU/XPU/CUDA)
- W&B or local experiment tracking

### Device Notes

| Device | Flag | Notes |
|--------|------|-------|
| CPU | `--device cpu` | Always available, slowest |
| Intel XPU | `--device xpu` | Requires `intel_extension_for_pytorch`; see [XPU_DIAGNOSIS.md](../reports/XPU_DIAGNOSIS.md) |
| CUDA | `--device cuda` | Standard NVIDIA GPU support |

## Registry Layout

Models are stored in a two-tier registry:

```
experiments/ml/
├── researcher/          # Researcher-trained models (manual, curated)
│   ├── throw_apex_v1/
│   │   ├── model.pt
│   │   ├── config.yaml
│   │   └── metrics.json
│   └── tower_stability_v2/
└── nn/                  # Agent-trained models (experimental, ephemeral)
    └── agent_001/
        └── throw_predictor/
```

*Per interface contract*:
- `researcher/` contains manually trained, version-controlled models
- `nn/` contains agent-initiated training runs
- Each model directory includes the saved weights, config, and evaluation metrics

## CLI Commands

*Per interface contract, section 3.5*, available commands:

```bash
# Dataset operations
playground ml generate --spec SPEC --output DIR    # Generate training data
playground ml inspect --data DIR                   # Inspect dataset statistics

# Training
playground ml train --spec SPEC --data DIR [OPTIONS]
playground ml evaluate --model MODEL --data DIR    # Run evaluation

# Registry
playground ml list                                 # List all models
playground ml show MODEL                           # Show model details
playground ml export MODEL --output FILE           # Export for deployment

# Device diagnostics
playground ml devices                              # Show available devices
```

## HTTP Endpoints

*Per interface contract, section 3.6*, the dashboard server exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ml/models` | GET | List registered models |
| `/api/ml/models/{id}` | GET | Get model details |
| `/api/ml/predict` | POST | Run inference with a model |
| `/api/ml/train` | POST | Start a training job |
| `/api/ml/jobs` | GET | List training jobs |
| `/api/ml/jobs/{id}` | GET | Get job status |

## Dashboard Tab

The ML Lab tab in the dashboard provides:
- Training job monitoring
- Model registry browser
- Evaluation visualization
- Device status

## Agent Learning Tools

*Per interface contract, section 3.3*, agents can:
- Request predictions from registered models
- Record training examples during sessions
- Trigger training runs on collected data

### Critical Rule: Predictions Are Never Evidence

**A neural network prediction does not constitute measured evidence.**

Agents must:
1. Use predictions only for planning (e.g., "the model suggests this throw might reach 3m")
2. Perform the actual action to obtain measured results
3. Record experiments citing the measured action, not the prediction

The evidence gate explicitly rejects:
- Experiments citing prediction results
- Discoveries based on model outputs without physical verification

## Neural Controller

The `neural` adapter runs a trained neural controller as an agent:

```bash
# Run a session with a neural controller
playground explore --agent agent_001 --adapter neural \
    --model researcher/policy_v1
```

*Per interface contract*, the neural adapter:
- Loads a policy network from the registry
- Maps observation to action at each step
- Records decisions for analysis

## Evaluation

Three evaluation modes ensure generalization:

### 1. Held-Out Evaluation
Standard train/test split from the same distribution.

```bash
playground ml evaluate --model MODEL --data test_split/
```

### 2. Fresh-Engine Evaluation
Test on physics runs not in the training set. Catches overfitting to specific state trajectories.

```bash
playground ml evaluate --model MODEL --mode fresh \
    --sessions NEW_SESSIONS
```

### 3. Closed-Loop Evaluation
Run the model as an actual agent and measure real outcomes. The only evaluation that catches distribution shift.

```bash
playground ml evaluate --model MODEL --mode closed_loop \
    --sessions 10 --agent neural_test
```

**Why closed-loop matters**: A model might predict throw heights accurately on held-out data but fail in practice because:
- The agent's approach path differs from training
- The model influences subsequent states
- Edge cases not in training data appear

## Known Limitations

1. **4B model exploration recorded no experiments**: The `lmstudio_unsloth4b` adapter was tested but did not produce experiment recordings during the evaluation period.

2. **Learning-from-memory not yet demonstrated**: While the infrastructure supports agent-driven training, no LLM-controlled agent has successfully completed the train→deploy→verify loop.

3. **Ollama Cloud returned 402**: External API limitations prevent using this adapter for extended experiments.

## See Also

- [ML_INTERFACE.md](ML_INTERFACE.md) - Technical interface specification
- [XPU_DIAGNOSIS.md](../reports/XPU_DIAGNOSIS.md) - Intel XPU debugging notes
- [OPERATING.md](OPERATING.md) - General operating procedures
