This is the official repository of our contribution at the
# <a href="https://frcsyn.github.io/" target="_blank">FRCSyn: Face Recognition Challenge in the Era of Synthetic Data</a>
which will be published in the proceedings of the <a href="https://wacv2024.thecvf.com/" target="_blank">WACV 2024 conference</a>.

<div align="center">
 <img src="./images/WACV-Logo_2024-1024x263.png"  width="750" alt="ROC curve Example"/> 
</div>
<br>
<div align="center">
 <img src="./images/intraclass.jpg"  width="750" alt="ROC curve Example"/> 
</div>


## Authors
<ul>
<li> Andrea Atzori (University of Cagliari) <a href="https://scholar.google.com/citations?hl=it&user=3_Z6fQ4AAAAJ" target="_blank">Google Scholar</a>
<li> Fadi Boutros (Fraunhofer IGD - TU Darmstadt) <a href="https://scholar.google.com/citations?user=C-zewBgAAAAJ&hl=it&oi=ao" target="_blank">Google Scholar</a>
<li> Naser Damer (Fraunhofer IGD - TU Darmstadt) <a href="https://scholar.google.com/citations?user=bAyT17sAAAAJ&hl=it&oi=ao" target="_blank">Google Scholar</a>
<li> Gianni Fenu (University of Cagliari) <a href="https://scholar.google.com/citations?user=riCjuhkAAAAJ&hl=it" target="_blank">Google Scholar</a>
<li> Mirko Marras (University of Cagliari) <a href="https://scholar.google.com/citations?user=JZhqKBIAAAAJ&hl=it&oi=ao" target="_blank">Google Scholar</a>
</ul>

We provide the Pytorch toolbox for Face Recognition training and testing that we used in the abovementioned challenge. 
It provides a training part with various Face Recognition backbones and losses and
an evaluation part that:
- Extracts the 512-sized embeddings from given datasets, by using previously trained FR models 
- Chooses the best possible threshold for FR verification task (using training-only info)
- Extracts system's verification binary decisions

## Requirements
- Python >= 3.8
- PyTorch >= 2.1.0
- DeepFace == 0.0.79
- MxNet == 1.9.0
- CUDA >= 12.0

In order to install all the necessary prerequisites, you can simply execute the following command: \
`pip install -r requirements.txt`

## Data Preprocessing
See <a href="src/1_preprocessing/README.md" target="_blank">README</a> in <a href="src/1_preprocessing" target="_blank">src/1_preprocessing</a> folder

## Model Training
See <a href="src/2_training/README.md" target="_blank">README</a> in <a href="src/2_training" target="_blank">src/2_training</a> folder


## Model Evaluation
See <a href="src/3_evaluation/README.md" target="_blank">README</a> in <a href="src/3_evaluation" target="_blank">src/3_evaluation</a> folder

## License

This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.



## Acknowledgements

We acknowledge financial support
under the National Recovery and Resilience Plan (NRRP),
Mission 4 Component 2 Investment 1.5 - Call for tender
No.3277 published on December 30, 2021 by the Italian
Ministry of University and Research (MUR) funded by
the European Union – NextGenerationEU. Project Code
ECS0000038 – Project Title eINS Ecosystem of Innovation
for Next Generation Sardinia – CUP F53C22000430001-
Grant Assignment Decree No. 1056 adopted on June 23,
2022 by the Italian Ministry of University and Research.
