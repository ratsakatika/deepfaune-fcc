#  WELCOME TO DEEPFAUNE SOFTWARE REPOSITORY


<img src="icons/logoINEE.png" width="50%" align=center>
<br>


---
# NEWS
---
## October 2025
Release v1.4 is available

* New categories 'golden jackal', 'raccoon dog', 'porcupine' and 'muskrat'.
* Bird classification into 'anseriform', 'columbiform', 'corvid', 'galliform', 'passerine', 'piciform', 'raptor', 'otherbird'  is possible (optional).
* New possibility to choose between detectors.
* Device choice is now possible.

Supported categories/species : BADGER, BEAR, BEAVER, BIRD, BISON, CAT, CHAMOIS, COW, DOG, EQUID, FALLOW DEER, FOX, GENET, GOAT, GOLDEN JACKAL, HEDGEHOG, IBEX, LAGOMORPH, LYNX, MARMOT, MICROMAMMAL, MOOSE, MOUFLON, MUSKRAT, MUSTELID, NUTRIA, OTTER, PORCUPINE, RACCOON, RACCOON DOG, RED DEER, REINDEER, ROE DEER, SHEEP, SQUIRREL, WILD BOAR, WOLF, WOLVERINE +  HUMAN + VEHICULE + EMPTY

+ (option) ANSERIFORM, COLUMBIFORM, CORVID, GALLIFORM, PASSERINE, PICIFORM, RAPTOR, OTHERBIRD


## February 2025
Release v1.3 is available.

* New categories 'bison', 'moose', 'reindeer' and 'wolverine'  (in french 'bison', 'elan', 'renne' and 'glouton').
* Even more efficient classification model, still based on vit_large_patch14_dinov2 architecture.
* New possibility to choose between our yolov8s at resolution 960 for detection and MegaDetector (Microsoft) yolov10x at resolution 640.
* Use of more icons instead of text in software design.
* Animal counts and human counts are managed independently, and displayed in the interface.
* Column 'HumanPresence' replaced by 'HumanCount'.

Supported categories/species : BADGER, BEAR, BEAVER, BIRD, BISON, CAT, CHAMOIS/ISARD, COW, DOG, EQUID, FALLOW DEER, FOX, GENET, GOAT, HEDGEHOG, IBEX, LAGOMORPH, LYNX, MARMOT, MICROMAMMAL, MOUFLON, MOOSE, MUSTELID, NUTRIA, OTTER, RACCOON, RED DEER, REINDEER, ROE DEER, SHEEP, SQUIRREL, WILD BOAR, WOLF, WOLVERINE + HUMAN + VEHICULE + EMPTY


## October 2024
Release v1.2 is available.  

Supported categories/species : BADGER, BEAR, BEAVER, BIRD, CAT, CHAMOIS/ISARD, COW, DOG, EQUID, FALLOW DEER, FOX, GENET, GOAT, HEDGEHOG, IBEX, LAGOMORPH, LYNX, MARMOT, MICROMAMMAL, MOUFLON, MUSTELID, NUTRIA, OTTER, RACCOON, RED DEER, ROE DEER, SHEEP, SQUIRREL, WILD BOAR, WOLF + HUMAN + VEHICULE + EMPTY 

---
# DOCUMENTATION & INSTALLATION PROCEDURE
---

Please refer to the online documentation at [https://deepfaune.pages.math.cnrs.fr/software/](https://deepfaune.pages.math.cnrs.fr/software/)

---
# LICENSE
---

All of the source code to this product is available under the [CeCILL](http://www.cecill.info), compatible with [GNU GPL](http://www.gnu.org/licenses/gpl-3.0.html).

Our model parameters ('deepfaune-*.pt' files) are available under the [Creative Commons Attribution-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-sa/4.0/).
They cannot be used without citing and referencing the name 'DeepFaune'.

Know your rights.

---
# TEAM & CONTACT
---

The DeepFaune software is developped by the Deepfaune team at CNRS. For more information about the project, please visit [https://www.deepfaune.cnrs.fr](https://www.deepfaune.cnrs.fr)

For any question, bug or feedback, feel free to send an email to [Vincent Miele](https://vmiele.gitlab.io/) <!--or use the Gitlab Service Desk-->

---
# REFERENCES
---

[Rig23] Rigoudy, N., Dussert G., the DeepFaune consortium, Spataro, B., Miele, V. & Chamaillé-Jammes, S. (2023) *The DeepFaune initiative: a collaborative effort towards the automatic identification of the European fauna in camera-trap images.* [European Journal of Wildlife Research](https://link.springer.com/article/10.1007/s10344-023-01742-7)

[Dus24] Dussert, G., Chamaillé-Jammes, S. Dray, S. &  Miele, V. (2024) *Being confident in confidence scores: calibration in deep learning models for camera trap image sequences.* [Remote Sensing in Ecology and Conservation](https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.412)

[Dus25] Dussert, G., Dray, S., Chamaillé-Jammes, S. &  Miele, V. (2025) *Paying Attention to Other Animal Detections Improves Camera Trap Classification Models.* [biorxiv](https://www.biorxiv.org/content/10.1101/2025.07.15.664849.full.pdf)