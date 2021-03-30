[toc]

# basicVSR_mge

A megengine implemention of basicVSR (CVPR 2021)

|           |                                                              |                                  |
| --------- | ------------------------------------------------------------ | -------------------------------- |
| megengine | Deep learning framework created by megvii, like pytorch, tensorflow etc... | https://megengine.org.cn/        |
| basicVSR  | BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond | https://arxiv.org/abs/2012.02181 |
|           |                                                              |                                  |

As of 2021.03.21, the **official implementation（https://github.com/ckkelvinchan/BasicVSR-IconVSR** has **not** been released. If the implementation of this repo is helpful to you, just star it, thanks!  

<img src="https://z3.ax1x.com/2021/03/21/64aQC6.png" alt="basicVSR" style="zoom: 33%;" />



<img src="https://z3.ax1x.com/2021/03/21/64aB28.png" alt="basicVSR" style="zoom: 33%;" />

# usage

## install

* Linux machine (you do not need to care about cuda version, only need NVIDIA graphics driver version greater than 418)
* python3.7
* `pip3 install megengine -f https://megengine.org.cn/whl/mge.html`
* `pip install -r requirements.txt `



## dataset preparation (REDS)

* link:    https://seungjunnah.github.io/Datasets/reds.html
* after unzip it ,you need to merge the training and validation dataset(like mmediting), thus total 270(240+30) clip, and remaining 30 clip for test.
* after merging,  your dir should like this:
  * train
    * train_sharp
      * 000
      * ...
      * 240 (the first validation clip, thus clip 000 of validation)
      * ...
      * 269
    * train_sharp_bicubic
      * X4
        * 000
        * ...
        * 269
  * test
    * test_sharp_bicubic
      * X4
        * 000
        * ...
        * 269

## Training

tutorial coming soon...



## Testing  (now only support REDS dataset)

### get checkpoints

use our trained model(generator_module.mge),  link:  https://drive.google.com/drive/folders/1MXwysTaBN-3qg-iHfxl0nkv6il0EcO2K?usp=sharing

**it has been trained 50 epochs on 240 clips, it's PSNR on validation dataset is 31.20(with 8* ensemble)**

### test on valid dataset

find the config file:  `configs/restorers/BasicVSR/basicVSR_test_valid.py` 

change the first three line for your situation

```
load_path = './workdirs/epoch_xxx'                               # must have generator_module.mge file in folder epoch_xxx, xxx is digital
dataroot = "pathtoyourdataset/train/train_sharp_bicubic"
exp_name = 'basicVSR_track1_test_for_validation'                # any name you like
```

and then , run  it:

```bash
cd xxx/basicVSR_mge
python  tools/test.py  configs/restorers/BasicVSR/basicVSR_test_valid.py --gpuids 0 -d
```

you can find the results in workdir

### test on test dataset

same to valid, to fix the config file `configs/restorers/BasicVSR/basicVSR_test_test.py`  first

```bash
cd xxx/basicVSR_mge
python  tools/test.py  configs/restorers/BasicVSR/basicVSR_test_test.py --gpuids 0 -d
```

> notice: only support one gpu config for gpuids now

## Results

* all output frames of test dataset produced by our model can be found here:  (3000 frames, trained only on 240 training clips)

https://drive.google.com/file/d/1r1TaTAltEocXNdHY8sOgTe5b9uRB2GqA/view?usp=sharing

