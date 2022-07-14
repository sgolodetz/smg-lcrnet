# smg-lcrnet

This Python package provides a wrapper around LCR-Net.

It is a submodule of [smglib](https://github.com/sgolodetz/smglib), the open-source Python framework associated with our drone research in the [Cyber-Physical Systems](https://www.cs.ox.ac.uk/activities/cyberphysical/) group at the University of Oxford.

### Installation (as part of smglib)

Note #1: Please read the [top-level README](https://github.com/sgolodetz/smglib/blob/master/README.md) for smglib before following these instructions.

Note #2: LCR-Net unfortunately requires PyTorch 0.4.0, and there are no corresponding binaries available for Python 3.7. Various fixes seem possible:
- Building PyTorch 0.4.0 from source against a later version of Python.
- Upgrading Detectron.pytorch to work with a later version of PyTorch. (This has been tried [here](https://github.com/adityaarun1/Detectron.pytorch), but this version seems to remove some things that LCR-Net needs.)
- etc.

However, these are all quite painful (much more painful than our chosen solution). To avoid the need to do any of them, we run LCR-Net in a separate process (as a service) in a Conda environment that uses Python 3.6.12.

1. Open the terminal.

2. Create a new Conda environment for LCR-Net, e.g.

   ```
   conda create -n lcrnet python==3.6.12 -c conda-forge
   ```

3. Active the Conda environment, e.g. `conda activate lcrnet`.

4. Install PyTorch 0.4.0 via:

   ```
   conda install https://anaconda.org/pytorch/pytorch/0.4.0/download/win-64/pytorch-0.4.0-py36_cuda90_cudnn7he774522_1.tar.bz2
   ```

5. Install the remaining dependencies of `Detectron.pytorch`:

   ```
   pip install cffi Cython numpy packaging pycocotools PyYAML torchvision==0.2.1
   ```

6. Change to the `<root>/smg-lcrnet/smg/external/lcrnet/Detectron.pytorch/lib` directory, and run `./make.sh` to build `Detectron.pytorch`.

7. Install the remaining dependencies of `smg-lcrnet`:

   i. First install `pygame` via `pip install pygame`.

   ii. Then install `smg-comms` and its supporting packages, in order, as per the relevant READMEs:

   - [smg-rigging](https://github.com/sgolodetz/smg-rigging/blob/master/README.md)
   - [smg-utility](https://github.com/sgolodetz/smg-utility/blob/master/README.md)
   - [smg-opengl](https://github.com/sgolodetz/smg-opengl/blob/master/README.md)
   - [smg-skeletons](https://github.com/sgolodetz/smg-skeletons/blob/master/README.md)
   - [smg-comms](https://github.com/sgolodetz/smg-comms/blob/master/README.md)

8. Download the LCR-Net model we use (`DEMO_ECCV18`) from [here](http://pascal.inrialpes.fr/data2/grogez/LCR-Net/pthmodels), and put it in a new directory called `models` beneath `<root>/smg-lcrnet/smg/external/lcrnet`.

9. Add the relevant directories to the Python path, e.g.

   ```
   export PYTHONPATH="C:/smglib/smg-lcrnet/smg/external/lcrnet/Detectron.pytorch/lib;$PYTHONPATH"
   export PYTHONPATH="C:/smglib/smg-lcrnet/smg/external/lcrnet;$PYTHONPATH"
   export PYTHONPATH="C:/smglib/smg-lcrnet;$PYTHONPATH"
   ```

10. You should then be able to successfully run `<root>/smg-lcrnet/scripts/run_lcrnet_skeleton_detection_service.py` (please test this).

---

Notes for PyCharm users:

- You can add the relevant directories to the Python path in the usual PyCharm way, i.e. by right-clicking on them and clicking ``Mark Directory as Sources Root``.

- It's nice (particularly in this case) to be able to use different Conda environments from within the same PyCharm project. See [here](https://stackoverflow.com/questions/37577785/multiple-python-interpreters-used-in-the-same-project/37578051) for how to set this up.

### Publications

If you build on this framework for your research, please cite the following paper:
```
@inproceedings{Golodetz2022TR,
author = {Stuart Golodetz and Madhu Vankadari* and Aluna Everitt* and Sangyun Shin* and Andrew Markham and Niki Trigoni},
title = {{Real-Time Hybrid Mapping of Populated Indoor Scenes using a Low-Cost Monocular UAV}},
booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
month = {October},
year = {2022}
}
```

### Acknowledgements

This work was supported by Amazon Web Services via the [Oxford-Singapore Human-Machine Collaboration Programme](https://www.mpls.ox.ac.uk/innovation-and-business-partnerships/human-machine-collaboration/human-machine-collaboration-programme-oxford-research-pillar), and by UKRI as part of the [ACE-OPS](https://gtr.ukri.org/projects?ref=EP%2FS030832%2F1) grant. We would also like to thank [Graham Taylor](https://www.biology.ox.ac.uk/people/professor-graham-taylor) for the use of the Wytham Flight Lab, [Philip Torr](https://eng.ox.ac.uk/people/philip-torr/) for the use of an Asus ZenFone AR, and [Tommaso Cavallari](https://uk.linkedin.com/in/tcavallari) for implementing TangoCapture.
