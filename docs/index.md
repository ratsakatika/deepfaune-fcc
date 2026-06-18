---
hide:
  - navigation # Hides the left-side navigation sidebar
---

# Welcome to DeepFaune documentation

DeepFaune is an initiative aiming at **providing AI models** to assist with the **automatic classification of species in camera-trap images or videos**. We also develop a cross-platform, desktop-based software to run the model with no computing expertise. You can find more information on DeepFaune and download the software here: [https://www.deepfaune.cnrs.fr](https://www.deepfaune.cnrs.fr)

This webpage provides documentation to help you install and use the software. We hope that you will find it useful!



## INSTALLATION

### For WINDOWS users

`DeepFaune` software is released under the [CeCILL](http://www.cecill.info) licence, compatible with [GNU GPL](http://www.gnu.org/licenses/gpl-3.0.html).

The latest versions are available at:
[https://pbil.univ-lyon1.fr/software/download/deepfaune/](https://pbil.univ-lyon1.fr/software/download/deepfaune/)

1. Download the latest `zip` file. If you have a `NVIDIA GPU`, please use the `*_gpu.zip` version.
2. Uncompress the file on your Desktop
3. Double-click on `deepfaune_installer.exe` to install the software on your computer

### For LINUX / MAC OS users (and WINDOWS users used to Python)

`DeepFaune` software is released under the [CeCILL](http://www.cecill.info) licence, compatible with [GNU GPL](http://www.gnu.org/licenses/gpl-3.0.html).

#### 1. Get the source code of the latest release, directly from this site. 

Option1 (latest version, recommended): clone the repository [https://plmlab.math.cnrs.fr/deepfaune/software/](https://plmlab.math.cnrs.fr/deepfaune/software/).

Option2 (latest stable version):  get the `zip` archive by clicking on the button `Download` (next to "Create release") on the last row of [https://plmlab.math.cnrs.fr/deepfaune/software/-/tags](https://plmlab.math.cnrs.fr/deepfaune/software/-/tags). Then, uncompress the zip file.

####  2. Get the model parameters

Our model parameters ('deepfaune-*.pt' files) are protected by the [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) (Attribution-ShareAlike 4.0 International).

Download the model parameters *inside the deepfaune folder* where you can find `deepfauneGUI.py`, if there are not yet present (they are in the zip file):

- Animal detector parameters: [deepfaune-yolov8s_960.pt](https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.4/deepfaune-yolov8s_960.pt)
and [md_v1000.0.0-sorrel.pt](https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.4/md_v1000.0.0.pt) courtesy of Dan Morris ([https://dmorris.net/](https://dmorris.net/)),
and possibly (optional) [md_v1000.0.0-redwood.pt](https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.4/md_v1000.0.0-redwood.pt) from MegaDetector. 

- Classifier parameters: [deepfaune-vit_large_patch14_dinov2.lvd142m.v4.pt](https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.4/deepfaune-vit_large_patch14_dinov2.lvd142m.v4.pt)
and [deepfaune-vit_large_patch14_dinov2.lvd142m.v4-bird_head.pt](https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.4/deepfaune-vit_large_patch14_dinov2.lvd142m.v4-bird_head.pt)

#### 3. Install the dependencies

##### Setting up your Python environment

On Linux, it can be recommended to create a virtual environement with:
```
python3 -m venv envdeepfaune
source env/bin/activatedeepfaune
pip install XXX
```

On Windows, Mac (& possibly Linux) it is recommended to use Anaconda:
```
conda create -n deepfaune
conda activate deepfaune
pip install XXX
```

On Windows, you can install these dependencies using **[Anaconda Individual Edition](https://www.anaconda.com/products/individual)**.

##### Dependencies

We need additional dependencies:

- PyTorch: `pip install torch torchvision`
- Yolov8: `pip install ultralytics`
- Yolov5: `pip install yolov5`
- Timm: `pip install timm`
- Pandas: `pip install pandas`
- Numpy: `pip install numpy`
- OpenCV: `pip install opencv-python`
- PIL: `pip install pillow`
- DILL: `pip install dill`
- hachoir: `pip install hachoir`
- (optional, for Excel users only) openpyxl: `pip install openpyxl`

For some users, it may be necessary to install `python-tk` or `python3-tk` as well, when you have a message `no module tkinter`...



## USING the DEEPFAUNE software

#### 0. Launch the software

On Windows, simply launch the software like any software. On Linux, open a terminal and type in the following command: python deepfauneGUI.py
Note that you can also call DeepFaune from your own script. Minimal examples are available in the `demo/` directory.

#### 1. Run an analysis on a set of images or videos

First, you need to let DeepFaune know where the medias to be analyzed are. Go to `File -> Import`, select either `Images` or `Videos`, and through the file explorer go to the folder containing the medias. Note that then the focus is on looking for a folder - your explorer will NOT display the medias within the folder. Click `OK` when you have reached the desired folder. The software will list your medias in the left column, and display the one selected.

Press `Configure & Run` at the bottom to start the analysis. A pop-up window will show up, in which you can adjust some configuration settings that will influence what results you get. We therefore recommend that you keep track of the configuration used for each run, in case you need to compare results later in the future. More details about the configuration settings are provided below. If using the default configuration settings, or once they have been adjusted, press `Run`. The software will run the detection and classification models. A progress bar indicating an approximate time to completion is shown at the bottom. Images are processed in batches and each time a batch is processed, the names of the images processed turn blue. Predictions for the medias processed are available immediately, even if some other medias have yet to be processed. Note that once you have launched a run, you cannot interrupt it.

#### Details and considerations on configuration settings:

- Choice of forbidden species

All species classes distinguished by the models are checked by default. This means that the model is allowed to classify a medias as containing any of these classes. In case you know that some species are not present in your study area, you can unchecked them. The model will then not be allowed to predict that a media contains an individual of these species. Unchecking species will often not be necessary, as the model usually perform well. However, doing it can be particularly useful when the model often confuses two species, one being absent from the study area. The software will remember your selection.

- Birds

We have trained a specific classification models to classify birds into coarse classes. This model is applied each time a media is classified as displaying by the global model. The class then becomes bird-subclass (e.g. bird-piciform). You can disable this option at each run.

- Confidence threshold under which to classify as `undefined`

Each media's classification is associated with a confidence score ranging from 0 to 1, with greater values indicating greater certainty in the prediction. We generally consider that, under a given confidence score threshold, the prediction is too uncertain and the medias should be inspected by a human. There is no objective way to define what this threshold should be, as it depends on the distribution of scores (which can change with each dataset), but more importantly depends to what extent you are happy to rely on the model's classifications, given the model's performance and your own use case (for some studies being nearly 100% is critical, for others not so much). By default, the threshold is set at 0.8, but you can modify it at each run. We recommend doing a number of test runs with different thresholds to identify what can be the optimal value for your use case and your data.

- Duration threshold

Camera-traps are often set-up to take a short sequence of pictures at each trigger. Also, sometimes several triggers follow each other over a short time interval, leading to longer sequences of pictures. In any case, when faced with a difficult image, human observers leverage the information contained in images collected right before or after. The software does the same, using an approach that we termed average-logit and that is explained in detail [here](https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.412). In approximate but simple terms, it averages predictions over the sequence to classify all images of the sequence at once, predicting the same class for all those images.
One however needs to decide what is the duration above which two consecutive images are considered to NOT be part of the same sequence. By default, we set this value at 10 seconds, but you can modify it at each run.


#### 2. Visualize & correct

Displaying a processed media, below the image you will see the class predicted, the associated confidence score, an ID number identifying the sequence (allowing you to check whether two medias were treated as being part of the same sequence or not). In addition, if you have activated the experimental option of counting animals (in menu Preferences), you will see the estimated number of animals and people. If an animal or a person has been detected, it will be framed in a red rectangle. If you have activated the human blurring option (in menu Preferences), parts of the images containing humans will now be blurred. This will also be the case in the images when you export them in subfolders (see below). Note that in some instances humans may fail to be classified as such by the model, and thus will not be blurred. We take no responsability for issues you could have because of this. It is your job to ensure you adhere to whatever regulations you are subject to.

Should you want to, you can correct the classifications within the software. Simply click on the predicted class, and select another one. You can even type in a new name if you wish, and that name will remain in the list of classes until you close the software.

#### 3. Export results, copy images

Once the images or videos have been analysed, Deepfaune users can export the results in `csv` or `xlsx` format. The file created contains a table with one row per image or video, and nine columns, the contents of which are explained below.

##### 3.1 Meaning of the columns of the results table
- `filename`: image path (including file name).
- `date`: date and time of the image. This field may not be filled in if the automatic import from the EXIF data has not been possible, which is always the case, for example, with videos in AVI format, which never contain the date and time.
- `seqnum`: numerical identifier of the image sequence to which this image belongs (i.e. images with the same number belong to the same sequence).
- `predictionbase`: class predicted by the model, without taking into account the other images of the sequence. This class is only reported in this field if the confidence score for this prediction (see scorebase below) is greater than the threshold set by the user. If this is not the case, the image is classified as `undefined`.
- `scorebase`: confidence score of the original prediction reported in predictionbase. This score can be interpreted as the probability that the prediction is correct.
- `prediction`: class predicted by the model, taking into account the other images of the sequence. As with the predictionbase field, this class is only reported in this field if the confidence score of this prediction (see score below) is greater than the threshold set by the user. If this is not the case, the image is classified as `undefined`.
- `score`: confidence score of the prediction reported in predictionbase. * top1: class predicted by the model, without taking into account the other images in the sequence. This class is always reported regardless of its confidence score. This field can be used by the user to assess whether the confidence score threshold could have been lowered while maintaining a correct classification, but reducing the number of images classified as 'undefined'.
- `humancount`: number of humans detected in this image, whatever the predicted class (for example, the predicted class could be `dog` but have a `humancount=1` if the image is of a person walking his dog).
- `count`: number of animals detected in this image (if the animal count option is enabled).


##### 3.2. Copy images into folders

The software also allows the user to copy the medias, once they have been analyzed by the software, into folders named following classes names. Just go to `File -> Copy`. The software allows to copy the raw images, or the images modified so that the humans detected by the software are blurred. This might be useful when sharing pictures with third-parties while wanted to keep some level of confidentiality.

#### 4. Additional settings

These options can be selected in the menu Preferences.

##### 4.1 Counting

When selected, the software will assume that all animals detected within one image are from the same species, and return the count of individuals. This option is selected disabled by default. This option only works for images, not for videos.

##### 4.2. Human blurring

When selected, the software will blur the bounding boxes in which a human would have been detected. The raw image is not altered, only the display in the software is blurred. If you copy the images with blurring, note that the copied version of images will be altered! This option is selected by default.

##### 4.3 Prediction for all bounding boxes

At the moment, a single bounding box is used per image. Further developments are in progress.

##### 4.4. Alternative detectors

We currently offer 3 alternative detectors, which vary in performance and speed. We recommend that users initially make a few test runs with each of them to select the one that is most suitable given the user's aims and constraints. Go to Models -> Detector to choose the one you want.

- Efficient and fast: we use our `deepfaune-yolov8s` as a first detector and use `MegaDetector v1000-sorrel` (courtesy of Dan Morris) as a backstop detector only for images classified as empty. Both models work at resolution `960 px`;

- More efficient but slower: we use an ensemble approach where we merge the results of the two detectors, our `deepfaune-yolov8s` and `MegaDetector v1000-sorrel` (courtesy of Dan Morris);

- Excellent but much slower: we use `MegaDetector v1000-redwood` proposed by the MegaDetector team; the model has a large number of parameters and works at resolution `1280 px`.



## USING the DEEPFAUNE API

You can implement your own scripts using the DeepFaune API. *Minimal examples* are available in the [demo/ directory](https://plmlab.math.cnrs.fr/deepfaune/software/-/tree/master/demo/).



## PERFORMANCE

We measured the performance (accuracy) of the classification model available in the latest stable release (1.4.0):

| Classes                      | Out-of-sample Test | Out-of-sample Test Support |
|------------------------------|--------------------|----------------------------|
| bison / bison                | 99,87%             | 4608                       |
| blaireau / badger            | 97,97%             | 5467                       |
| bouquetin / ibex             | NA                 | 0                          |
| castor / beaver              | 81,82%             | 11                         |
| cerf / red deer              | 96,20%             | 80393                      |
| chacal doré / golden jackal  | 93,35%             | 2877                       |
| chamois / chamois            | 96,25%             | 5674                       |
| chat / cat                   | 96,59%             | 20481                      |
| chevre / goat                | 74,33%             | 1239                       |
| chevreuil / roe deer         | 97,88%             | 24786                      |
| chien / dog                  | 97,47%             | 2218                       |
| chien viverrin / raccoon dog | 91,21%             | 700                        |
| daim / fallow deer           | 95,82%             | 717                        |
| ecureuil / squirrel          | 96,20%             | 2131                       |
| elan / moose                 | 97,33%             | 6391                       |
| equide / equid               | 92,42%             | 2978                       |
| genette / genet              | 98,47%             | 2094                       |
| glouton / wolverine          | 81,99%             | 272                        |
| herisson / hedgehog          | 94,12%             | 51                         |
| lagomorphe / lagomorph       | 98,83%             | 17336                      |
| loup / wolf                  | 98,68%             | 152                        |
| loutre / otter               | 100,00%            | 2                          |
| lynx / lynx                  | 99,81%             | 1047                       |
| marmotte / marmot            | 98,32%             | 1488                       |
| micromammifere / micromammal | 96,90%             | 258                        |
| mouflon / mouflon            | 82,28%             | 711                        |
| mouton / sheep               | 99,09%             | 6560                       |
| mustelide / mustelide        | 96,19%             | 4989                       |
| oiseau / bird                | 99,53%             | 39345                      |
| ours / bear                  | 97,03%             | 3740                       |
| porc-épic / porcupine        | 97,25%             | 581                        |
| ragondin / nutria            | 92,73%             | 770                        |
| rat musqué / muskrat         | 75,00%             | 25                         |
| ratonlaveur / racoon         | 87,51%             | 8234                       |
| renard / fox                 | 96,08%             | 31915                      |
| renne / reindeer             | 98,84%             | 518                        |
| sanglier / wild boar         | 98,66%             | 113814                     |
| vache /cow                   | 96,43%             | 7647                       |

And for the bird classification model :

| Classes     | Out-of-sample Test | Out-of-sample Test Support |
|-------------|--------------------|----------------------------|
| anseriform  | NA                 | 0                          |
| columbiform | 100,0%             | 6                          |
| corvid      | 96,55%             | 29                         |
| galliform   | 100,0%             | 41                         |
| passerine   | 99,22%             | 4095                       |
| piciform    | 100,0%             | 22                         |
| raptor      | 98,57%             | 70                         |
| otherbird   | 97,96%             | 3673                       |



## FREQUENTLY ASKED QUESTIONS

> Is the `DeepFaune` software free?

Yes, it is a free software (see LICENSE section). If you appreciate our work, please cite our work and/or contribute by sharing with us your annotated images.

> Can I have access to the images used in the DeepFaune project?

No. We do not share the images of our partners.

> Can I contribute to the DeepFaune project with my images?

It would be great!! You can contact us to see how you can send us your images (we have different solutions). We will store them in a secure server with private access to the members of the deepfaune project.
 
> Who is developing this DeepFaune project?

A CNRS team led by Simon Chamaillé-Jammes (CEFE), Gaspard Dussert (LBBE)  and Vincent Miele (LECA). Please have a look at our website [https://www.deepfaune.cnrs.fr/](https://www.deepfaune.cnrs.fr/).

