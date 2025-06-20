# GenDSA-V2


## <img src='/sundry/1f9f3.gif' width="30px"> Environment Setups

* python 3.8
* cudatoolkit 11.2.1
* cudnn 8.1.0.77
* See 'GenDSA_env.txt' for Python libraries required

```shell
conda create -n GenDSA python=3.8
conda activate GenDSA
conda install cudatoolkit=11.2.1 cudnn=8.1.0.77
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# cd /xx/xx/GenDSA
pip install -r GenDSA_env.txt
```


## <img src='/sundry/1f5c2-fe0f.gif' width="30px"> Model Checkpoints
Download the [model checkpoints](https://drive.google.com/drive/folders/1lB0jEF581p5csDq1VzLhjCCaIVCBXmx_?usp=sharing), put all pkl files into ../GenDSA/weights/checkpoints.

## <img src='/sundry/听诊器.gif' width="30px"> Our Dataset and Inference Cases
We released a portion of the [retrospective data.](https://github.com/ZrH42/GenDSA_Data)


## <img src='/sundry/1f3ac.gif' width="30px"> Inference Demo
Run the following commands to generate multi-frame interpolation:

* Two-frame interpolation
```shell
python Simple_Interpolator.py \
--model_path ./weights/checkpoints/3D-Head-Inf2.pkl \
--frame1 ./demo_images/DSA_1.png \
--frame2 ./demo_images/DSA_2.png \
--inter_frames 2
```

* Three-frame interpolation
```shell
python Simple_Interpolator.py \
--model_path ./weights/checkpoints/3D-Head-Inf3.pkl \
--frame1 ./demo_images/DSA_1.png \
--frame2 ./demo_images/DSA_2.png \
--inter_frames 3
```

* Two-frame 3D_Head sequence interpolation

```shell
python ImageSequenceInterFrame.py \
--model_path ./weights/checkpoints/3D-Head-Inf2.pkl \
--input_folder ./demo_sequences/3D_Head \
--output_folder ./demo_sequences/3D_Head_Inter2 \
--inter_frames 2 \
--start_frame 15
```

* Three-frame 2D_Head sequence interpolation

```shell
python ImageSequenceInterFrame.py \
--model_path ./weights/checkpoints/2D-Head-Inf3.pkl \
--input_folder ./demo_sequences/2D_Head \
--output_folder ./demo_sequences/2D_Head_Inter3 \
--inter_frames 3 \
--start_frame 15
```

You can also use other checkpoints to generate 1~3 frame interpolation for your 2D/3D - Head/Abdomen/Thorax/Pelvic/Periph images.

## 💖 Citation
Please promote and cite our work if you find it helpful. Enjoy!
```shell
@article{zhao2024large,
    title={Large-scale pretrained frame generative model enables real-time low-dose DSA imaging: An AI system development and multi-center validation study},
    author={Zhao, Huangxuan and Xu, Ziyang and Chen, Lei and Wu, Linxia and Cui, Ziwei and Ma, Jinqiang and Sun, Tao and Lei, Yu and Wang, Nan and Hu, Hongyao and others},
    journal={Med},
    year={2024},
    publisher={Elsevier},
    url={https://doi.org/10.1016/j.medj.2024.07.025},
}
```
