# Image Segmentation Experiment Combining Segmentation Models and QWEN-VL-Chat Model

This repository contains the source code, documentation, and materials associated with the research project "Image Segmentation Experiment Combining Segmentation Models and QWEN-VL-Chat Model" presented by Guo Chumeng from The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, Guangdong province, China.

The whole experiment of this project mainly contains three parts.
- **Reproducing Qwen-VL**
-  **Image Segmentation Using SAM**
-  
## Overview

The project explores the integration of advanced multimodal models to enhance image content comprehension and intelligent interaction. It builds upon the success of models like LLaVA and QWEN-VL in tasks such as visual chat, image editing, and object detection. The main goal is to construct an intelligent system capable of dynamically editing and segmenting images during dialogue processes, providing users with a more intuitive and interactive experience.

## Key Contributions

1. **Integration of Qwen-VL and Segment-Anything Project**: The innovative approach of integrating the Qwen-VL-Chat model with the Segment-Anything project enables deep understanding and segmentation of objects within images.

2. **Development of an Integrated System**: A novel system that combines QwenVL’s conversational capabilities with Segment-Anything’s image segmentation functionality, achieving comprehensive image understanding and editing in real-time.

3. **Dynamic Response Mechanism**: Incorporation of a dynamic response mechanism within Qwen-VL to detect specific tags in the generated responses and dynamically invoke the segmentation model.

4. **Enhanced User Experience**: Advanced AI models integration to provide intelligent image understanding and editing capabilities, enhancing user interaction with the system.

## System Design and Methodology

The system design involves the following key components:

- **Reproducing Qwen-VL**: Replication of the functionalities of the Qwen-VL and Qwen-VL-Chat models for tasks involving image descriptions and image-related conversational generation.

- **Image Segmentation Using SAM**: An innovative approach that combines the Qwen-VL-Chat model with the Segment-Anything project for object detection and segmentation tasks in images.

- **Image Segmentation Using GroundingDino+SAM**: A method that allows segmentation of objects directly during the Q&A process with the multimodal model.

- **Processing and Alignment of the Fine-Tuning Dataset**: Utilization of the dataset `lava-plus-v1-117ktool-merge.json` for fine-tuning the Qwen-VL-Chat model.

- **Fine-tuning**: Fine-tuning of the Qwen-VL-Chat model using a single GPU training method with LoRA.

## Usage

To reproduce the experiments and utilize the integrated system, follow these steps:

1. **Environment Setup**: Ensure the specified environment requirements are met, including Python version 3.10, PyTorch version 1.12 or higher, and the appropriate CUDA version.

2. **Code Access**: Access the code from the Qwen-VL project page on GitHub and download the required model files from the Hugging Face Model Hub.

3. **Model and Code Preparation**: Load the necessary model and tokenizer using the Transformers library, setting appropriate parameters for generation.

4. **Inference**: Conduct inference using given example dialogues to test the integrated system's capabilities.

5. **Fine-tuning**: Perform fine-tuning of the Qwen-VL-Chat model with the prepared dataset and parameters.

6. **Experimentation**: Run the experiments using the provided scripts and observe the system's performance in image segmentation and dialogue interaction.

## Future Work

- Comprehensive evaluation of the system's performance, usability, and robustness.
- Integration of other large language models to improve question answering and instruction understanding.
- Development of a more dynamic and interactive dialogue system for real-time feedback.
- In-depth research on multimodal fusion technology for effective integration of visual and language information.

## References

The project references several key papers and resources, detailed in the provided documentation. For more information, refer to the `References` section in the documentation.

## License

This project is open-source under [LICENSE NAME], allowing for modification and redistribution under certain conditions. Please review the license for more details.

## Contact

For inquiries, feedback, or collaboration, please contact Guo Chumeng at [cguo847@connect.hkust-gz.edu.cn](mailto:cguo847@connect.hkust-gz.edu.cn).
