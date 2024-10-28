**Official "Image Inpainting via Multi-scale Adaptive Priors" code implemented with pytorch.
The pretrained models for datasets CelebA [liu2015deep], Places2 [zhou2017places], and Paris StreetView [doersch2015makes] under irregular masks [liu2018image] are available in "[Google Drive](https://drive.google.com/drive/folders/1cZ2E5uzGqDp5X0ICphewGVbP4kwMlsxj?usp=sharing)". You can directly load and test them.**

@inproceedings{liu2015deep,
  title={Deep learning face attributes in the wild},
  author={Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={3730--3738},
  year={2015}
}

@article{zhou2017places,
  title={Places: A 10 million image database for scene recognition},
  author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={40},
  number={6},
  pages={1452--1464},
  year={2017},
  publisher={IEEE}
}

@article{doersch2015makes,
  title={What makes paris look like paris?},
  author={Doersch, Carl and Singh, Saurabh and Gupta, Abhinav and Sivic, Josef and Efros, Alexei A},
  journal={Communications of the ACM},
  volume={58},
  number={12},
  pages={103--110},
  year={2015},
  publisher={ACM New York, NY, USA}
}

@inproceedings{liu2018image,
  title={Image inpainting for irregular holes using partial convolutions},
  author={Liu, Guilin and Reda, Fitsum A and Shih, Kevin J and Wang, Ting-Chun and Tao, Andrew and Catanzaro, Bryan},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={85--100},
  year={2018}
}

**In our comparative experiments, certain methods did not provide official pretrained models for the given dataset and mask type. To ensure a fair comparison, we retrained these models based on the officially stated training pipelines under our experimental setup. We have also released the retrained weights, their are**: 

MI-GAN in Paris StreetView: "[Google Drive]([https://drive.google.com/drive/folders/1cZ2E5uzGqDp5X0ICphewGVbP4kwMlsxj?usp=sharing](https://drive.google.com/drive/folders/1o9BkRxIEFfrkP2cCAqA2qPHD_02XmBvP))"

MI-GAN in Places2: "[Google Drive]([https://drive.google.com/drive/folders/1cZ2E5uzGqDp5X0ICphewGVbP4kwMlsxj?usp=sharing](https://drive.google.com/drive/folders/1o9BkRxIEFfrkP2cCAqA2qPHD_02XmBvP))"

MI-GAN in CelebA: "[Google Drive]([https://drive.google.com/drive/folders/1cZ2E5uzGqDp5X0ICphewGVbP4kwMlsxj?usp=sharing](https://drive.google.com/drive/folders/1o9BkRxIEFfrkP2cCAqA2qPHD_02XmBvP))"

MISF in Paris StreetView: "[Google Drive]([https://drive.google.com/drive/folders/1cZ2E5uzGqDp5X0ICphewGVbP4kwMlsxj?usp=sharing](https://drive.google.com/drive/folders/1o9BkRxIEFfrkP2cCAqA2qPHD_02XmBvP))"

