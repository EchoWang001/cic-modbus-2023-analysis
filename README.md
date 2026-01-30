# Modbus Anomaly Detection Framework

This repository contains the source code and experimental framework for anomaly detection in Modbus-based Industrial Control Systems (ICS). The project is based on the **CIC Modbus Dataset 2023** and focuses on feature-based detection using machine learning.

## Project Overview

Modbus is a widely used industrial communication protocol that lacks built-in security features. This project provides a comprehensive framework for:
1.  **PCAP Parsing**: Efficiently extracting Modbus application layer fields from raw network captures.
2.  **Feature Engineering**: Generating 44 distinct features across Protocol, Temporal, and Operational Pattern categories.
3.  **Anomaly Detection**: Implementing and evaluating multiple machine learning models (Random Forest, XGBoost, MLP, etc.) for real-time threat detection.

## Key Features

- **High-Performance Parsing**: Built on Scapy for robust packet analysis.
- **Advanced Feature Set**: Includes specialized DCS business logic features like `consecutive_write_max` and `write_without_read_ratio`.
- **Lightweight Implementation**: Designed for potential deployment on resource-constrained IIoT edge devices.
- **Comprehensive Evaluation**: Includes scripts for ablation studies, cross-scenario generalization, and computational efficiency testing.

## Project Structure

```text
.
├── data/               # Dataset directory (raw, processed, splits)
├── docs/               # Project design and summary documents
├── models/             # Saved model weights and metadata
├── notebooks/          # Jupyter notebooks for experimental analysis
├── paper/              # LaTeX source for the research paper
├── results/            # Experimental results (tables and figures)
├── src/                # Core source code
│   ├── config.py           # Project configuration and paths
│   ├── pcap_parser.py      # PCAP to packet-level CSV parser
│   ├── feature_extractor.py # Window-level feature extraction logic
│   └── full_sampling.py    # Large-scale data sampling script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.9+
- [Conda](https://docs.conda.io/en/latest/) (recommended)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/modbus-detection.git
    cd modbus-detection
    ```

2.  Create and activate the environment:
    ```bash
    conda create -n modbus-detection python=3.9
    conda activate modbus-detection
    pip install -r requirements.txt
    ```

3.  Configure the dataset path in `src/config.py`:
    ```python
    DATASET_ROOT = "/path/to/your/CIC_Modbus_Dataset_2023"
    ```

### Usage

1.  **Data Preprocessing**:
    Run the sampling script to parse PCAPs and align labels:
    ```bash
    python src/full_sampling.py
    ```

2.  **Feature Extraction & Training**:
    Follow the steps in `notebooks/08_Phase2_基础实验.ipynb` to extract features and train models.

3.  **Evaluation**:
    Use `notebooks/09_Phase3_核心验证.ipynb` for detailed performance analysis and ablation studies.

## Citation

If you use this code or framework in your research, please cite our paper:

> W. Yujie and W. Zhaohui, "Feature-based Anomaly Detection in Modbus Networks: A Comprehensive Study on CIC Dataset 2023," in Proc. IEEE 24th Int. Conf. Industrial Informatics (INDIN), Melbourne, Australia, 2026.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Canadian Institute for Cybersecurity (CIC) for providing the [Modbus Dataset 2023](https://www.unb.ca/cic/datasets/modbus-2023.html).
- IEEE Industrial Electronics Society (IES) for the INDIN conference platform.
