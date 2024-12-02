# GenAug

## Installation

Clone GenAug repo:
```bash
git clone https://github.com/wxinhua/genaug.git
```
Install required packages:
```bash
conda create -n Genaug python=3.10
conda activate Genaug
cd genaug
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install -r requirements.txt
```
需要的模型有：
* GroundingDINO
* SAM
* Stable_Diffusion_2_Inpainting
* Stable-Diffusion-2-Depth

## Quickstart
需要的输入有：原h5文件路径（到数字文件夹前），输出路径，需要保留的object text prompt,模型路径

输出增强后的h5文件
```bash
bash back2h5.sh input_path(~/success_episodes) output_path file_number(xxxxxx) augment_time item 
```



## Citations
**GenAug**
```bibtex
@article{chen2023genaug,
  title={GenAug: Retargeting behaviors to unseen situations via Generative Augmentation},
  author={Chen, Zoey and Kiami, Sho and Gupta, Abhishek and Kumar, Vikash},
  journal={arXiv preprint arXiv:2302.06671},
  year={2023}
}
```

**Stable Diffusion**
```bibtex
@inproceedings{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10684--10695},
  year={2022}
}
```
**TransporterNet**
```bibtex
@inproceedings{zeng2020transporter,
  title={Transporter networks: Rearranging the visual world for robotic manipulation},
  author={Zeng, Andy and Florence, Pete and Tompson, Jonathan and Welker, Stefan and Chien, Jonathan and Attarian, Maria and Armstrong, Travis and Krasin, Ivan and Duong, Dan and Sindhwani, Vikas and others},
  booktitle={Proceedings of the 4th Conference on Robot Learning (CoRL)},
  year= {2020},
}
```
**CLIPort**
```bibtex
@inproceedings{shridhar2021cliport,
  title     = {CLIPort: What and Where Pathways for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 5th Conference on Robot Learning (CoRL)},
  year      = {2021},
}
```
## Questions or Issues?

Please file an issue with the issue tracker.  