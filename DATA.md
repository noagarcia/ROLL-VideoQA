### Data Preparation

ROLL is designed to leverage external information to answer knowledge-based questions about videos. 
We reported experiments on two datasets: [KnowIT VQA](https://knowit-vqa.github.io/) and the [TVQA+](http://tvqa.cs.unc.edu/download_tvqa_plus.html). 
Both datasets contain videos from the Big Bang Theory, so list of characters and common locations are shared.


### KnowIT VQA
1. Download annotations from [here](https://knowit-vqa.github.io/) and extract the zip file contents into `Data/` directory. 
You should get 3 csv files inside `Data/knowit_data/`.
2. The episode summaries used as external knowledge are in `Data/knowledge_base/tbbt_summaries.csv`. 
3. The video story identification has been already pre-computed and can be found in `Data/knwoledge_base/`.
4. For the Observe Branch we provide the pre-computed visual features: 
    - [Download](https://drive.google.com/open?id=1DuIuXBJuO47JkZYXROOGB1q0XncxF5Lt) zip file with pre-computed characters, places, and actions. 
    Extract the three files in `Data/knowit_observe/` directory.
    - [Object relations]() zip file (146.8 GB) with the pre-computed object relations extracted with [VRD](https://github.com/facebookresearch/Large-Scale-VRD).
    
The final `Data/` directory structure should look like this:
``` 
Data
|_ knowit_data
|  |_ knowit_data_test.csv
|  |_ knowit_data_train.csv
|  |_ knowit_data_val.csv
|_ knowit_observe
|  |_ knowit-vrd
|  |  |_  s01e01.pkl
|  |  |_  s01e02.pkl
|  |  |_  ...
|  |_ knowit_action_predictions.pkl
|  |_ knowit_character_recognition.tsv
|  |_ knowit_places_classification.csv 
|_ knowledge_base
|  |_ retrieved_episode_from_scenes_test.csv
|  |_ retrieved_episode_from_scenes_train.csv
|  |_ retrieved_episode_from_scenes_val.csv
|  |_ tbbt_summaries.csv
|_ actions_charades_classes.txt
|_ actions_framelist.csv
|_ vg_objects.json
|_ vg_predicates.json

```

### TVQA+
TODO.

