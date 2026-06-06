# UM Cryo-ET workshop: Processing heterogeneous data with tomoDRGN and SIREn  

**Presenter**: Laurel Kinman  
**Date**: June 10, 2026  
  
  
## Overview  
  
This tutorial will walk you through processing an *in situ* dataset of ribosomes from *M. pneumoniae* cells treated with chloramphenicol ([EMPIAR-10499](https://www.ebi.ac.uk/empiar/EMPIAR-10499/)). We will use [tomoDRGN](https://www.nature.com/articles/s41592-024-02210-z), a deep learning tool for analyzing heterogeneous cryo-ET datasets, to first filter an inpute particle stack and then use the filtered stack to reconstruct the conformational states sampled by the ribosome in this dataset. Finally, we will use [SIREn](https://github.com/lkinman/siren) for unbiased identification of heterogeneous regions in these conformational states.   

For more information on running tomoDRGN, you can refer to the helpful [GitPages documentation](https://bpowell122.github.io/tomodrgn/index.html), which I have based parts of this tutorial on.  

This tutorial will involve a mix of commands you will need to run, and pre-calculated results that we will look at and analyze together. The following flags are used throughout the text to highlight stopping points or further exercises you can explore on your own:  

>:triangular_flag_on_post: indicates a moment for us to pause and come together as a group to discuss the results we are seeing. Stop here!   

>:white_check_mark: indicates something that might be useful for you to follow up with on your own  
  
  
## Contents  
  
-[Tutorial data](#tutorial-data)  
-[Validating particle extraction](#validating-particle-extraction)  
-[Training an initial tomoDRGN model](#training-an-initial-tomodrgn-model)  
-[Filtering the input particle stack](#filtering-the-input-particle-stack)  
-[Training a final tomoDRGN model](#training-a-final-tomodrgn-model)  
-[Identifying variable regions with SIREn](#identifying-variable-regions-with-siren)  
-[Measuring the occupancy of SIREn blocks](#measuring-the-occupancy-of-siren-blocks)  
-[Selecting particles based on SIREn block occupancy](#selecting-particles-based-on-siren-block-occupancy)
-[Refining SIREn results with a filtered volume ensemble](#refining-siren-results-with-a-filtered-volume-ensemble) 
-[References](#references)  
-[Appendix 1: Custom Warp build](#appendix-1-custom-warp-build)  
-[Appendix 2: Particle preprocessing pipeline](#apppendix-2-particle-preprocessing-pipeline)  
   
    
## Tutorial data  
  
For this tutorial, we are using a preprocessed stack of particles extracted from the [EMPIAR-10499](https://www.ebi.ac.uk/empiar/EMPIAR-10499/) dataset. These particles were processed with WarpTools v2.0, [Miss-Alignment](https://www.biorxiv.org/content/10.64898/2026.04.29.721716v1) for tilt-series alignment, and [RELION-5](https://febs.onlinelibrary.wiley.com/doi/10.1002/2211-5463.13873) for 3D refinement.  
  
**Important note:** These particles were processed with a custom build of Warp that allowed export of particles images that were not pre-multiplied by the CTF. For notes on how to build this version of Warp, please refer to [Appendix 1](#appendix-1-custom-warp-build). A complete preprocessing pipeline is provided in [Appendix 2](#apppendix-2-particle-preprocessing-pipeline). Processing these particles as described below also requires tomoDRGN v1.0.3+.  
  
tomoDRGN takes as input 2D particle image series. We also recommend exporting 3D subtomograms for some of the downstream analysis steps, however. Here, you are provided with both. 
  
  
[back to top](#um-cryo-et-workshop-processing-heterogeneous-data-with-tomodrgn-and-siren)  
  
  
## Validating particle extraction  
  
Typically, we recommend validating particle extraction and tomoDRGN preprocessing using either ```tomodrgn backproject_voxel``` or  ```tomodrgn train_nn```, which trains a decoder-only homogeneous structural model from the particle images.  
  
Before we do that, we need to activate our tomoDRGN conda environment. We're also going to locate ourselves in our scratch directory, and copy over some of the relevant data. In a terminal window, run the following commands:  
  
```
source ~/conda_init.sh
conda activate tomodrgn
cd /scratch
cp /work/data/EMPIAR-10499/relion/Refine3D/job009/run_optimisation_set.star ./
```
  
You should now be located in ```/scratch```, which should contain a new file ```run_optimisation_set.star```. The last thing we will have to do before we can proceed with validating particle extraction is fixing the relative paths in this file. To do so, just open ```run_optimisation_set.star``` in vi or a text editor of your choice. Prepend ```/work/data/EMPIAR-10499/relion``` to each of the two paths you see listed. 
  
We are now prepared to run ```backproject_voxel``` using the following command:
  
``` 
tomodrgn backproject_voxel run_optimisation_set_fixedpaths.star --output 00_backproject/backproject_weighted.mrc --recon-dose-weight --recon-tilt-weight --source-software warptools --image-dose-weighted false --image-ctf-premultiplied false --datadir /work/data/EMPIAR-10499/warp_tiltseries/particleseries/
```
  
>:triangular_flag_on_post: What is going on in this command? Let's break it down together. 
  
While this runs, check out the options you can pass to ```backproject_voxel``` by running
  
```
tomodrgn backproject_voxel --help
```  
   
When running this command, the primary thing to look out for is volumes that look hollow or otherwise do not produce an interpretable (if noisy or low resolution) structure that resembles the 3D refinement. If you observe these features in your ```backproject_voxel``` or ```train_nn``` map, try re-running the command with ```--uninvert-data```. By default, tomoDRGN assumes data is light-on-dark and needs to be inverted before training. If your data is dark on light, you will need to provide ```--uninvert-data``` to ```backproject_voxel```, ```train_nn```, and ```train_vae``` in order to get interpretable maps.  
  
To get an intuition for what you might see if you get the sign convention wrong, try running ```backproject_voxel``` again, this time providing ```--uninvert-data```.


```
tomodrgn backproject_voxel run_optimisation_set.star --output 01_backproject_wrong/backproject_weighted.mrc --recon-dose-weight --recon-tilt-weight --source-software warptools --image-dose-weighted false --image-ctf-premultiplied false --uninvert-data --datadir /work/data/EMPIAR-10499/warp_tiltseries/particleseries/
```  
  
    
[back to top](#um-cryo-et-workshop-processing-heterogeneous-data-with-tomodrgn-and-siren)  
  
  
## Training an initial tomoDRGN model  
  
Once we’re confident with upstream particle extraction and processing, we can train a tomoDRGN model using ```train_vae```. First, we can examine the command-line options available to us using:

```
tomodrgn train_vae --help
```  
  
This returns a lot of options! Most of them will not be particularly relevant to us outside of specialized use cases.    
  
The ones you should pay attention to are:
-  ```-o``` specifies an output directory to store the results in. I like to give this an informative name
- ```-n``` indicates the total number of epochs you will train for. You can always analyze epochs before the final one, or pick up and train for more epochs later.
- ```-b``` specifies the desired GPU minibatch size. If you change this from 8, you may need to play with the learning rate to optimize results. 
- ```--zdim``` The dimensions of the final (post- encoder B) latent embedding
- ```--source-software``` tells tomoDRGN how to parse the input .star file. If you don't provide this, tomoDRGN will try to autodetect the software (and may fail)
- ```--recon-dose-weight```, ```--recon-tilt-weight```, and ```--l-dose-mask``` handle tilt and dose-weighting intelligently, you will pretty much always want to include these
- ```--num-workers``` will largely be determined by your compute setup  
- ```--uninvert-data```, as described above  
  
Now, we will put together the training command for our dataset.  
  
```
tomodrgn train_vae \
    run_optimisation_set.star \
    -o 02_4Apix_box96_z128_b8 \
    -n 50 \
    -b 8 \
    --zdim 128 \
    --recon-dose-weight \
    --recon-tilt-weight \
    --l-dose-mask \
    --image-dose-weighted false \
    --image-ctf-premultiplied false \
    --source-software warptools \
    --num-workers 2 \
    --prefetch-factor 2 \
    --pin-memory\
    --datadir /work/data/EMPIAR-10499/warp_tiltseries/particleseries/
```

>:triangular_flag_on_post: What do you want to keep an eye on throughout training? How do you monitor the status of the trained model?  

Once this model has finished training, you can analyze the outputs using ```tomodrgn analyze```. Remember that you can (and should!) analyze the outputs at multiple epochs, to ensure that training has converged and to prevent overfitting. Automated analysis of model convergence can also be carried out using ```tomodrgn convergence_vae```.

For the sake of consistency, we're going to analyze pre-calculated results together, located in ```/work/data/EMPIAR-10499/tomodrgn/18_4Apix_box96_z128_b8_v103```. This model was trained using the same parameters and particle stacks you used. We'll start by copying that directory over to scratch, and then analyzing the outputs of tomodrgn across a couple of selected epochs. 

```
cp -r /work/data/EMPIAR-10499/tomodrgn/18_4Apix_box96_z128_b8_v103 ./
tomodrgn analyze 18_4Apix_box96_z128_b8_v103 --epoch 4 --ksample 20
tomodrgn analyze 18_4Apix_box96_z128_b8_v103 --epoch 24 --ksample 20
tomodrgn analyze 18_4Apix_box96_z128_b8_v103 --epoch 49 --ksample 20
```
  
Take a moment to familiarize yourself with the outputs of ```tomodrgn analyze```.  
  
>:triangular_flag_on_post: What are the useful plots to consult when trying to understand how tomoDRGN has performed on our dataset? What does each plot depict?     
  
For each epoch analyzed, open the 20 volumes generated by k-means clustering of the latent space in ChimeraX. Starting with the volumes from epoch 4, make some notes on what you see. Do the volumes look like ribosomes? Do you see any evidence of 'junk' particles? What about more interesting forms of structural heterogeneity?  
  
>:triangular_flag_on_post: What do you notice in the k-means volumes? Are everyone's k-means volumes the same? How do the k-means volumes change across epochs?  
  
>:white_check_mark: When your own model finishes training, you can run ```tomodrgn analyze``` on those results and compare them to the precalculated results.  
    
  
[back to top](#um-cryo-et-workshop-processing-heterogeneous-data-with-tomodrgn-and-siren)  
  
  
## Filtering the input particle stack  
  
In the results of our first tomoDRGN model, we find evidence that the particle stack was imperfectly filtered and contains residual junk or non-ribosomal particles. This makes sense, since this particle stack was subjected to only a single round of 3D classification before refinement. Junk particles may consume some of the network's representation capacity and reduce the ability of the network to reconstruct more interesting structural heterogeneity. So we are going to filter the particle stack, and then train a new tomoDRGN model using the filtered particle stack.  

First, let's take a look at the tomoDRGN command we can use to filter particle stacks:

```
tomodrgn filter_star --help
```
  
As you can see, a number of filtering methods are available directly through ```filter_star```. The script allows you to either keep or drop particles (using ```--action```), either from selected individual tomograms, using provided particle indices, or based on provided labels. 
  
Here, we have a relatively clear set of particles we want to omit just from the k-means clustering with **k** = 20 at 49 epochs. To do that filtering, you will first need to delete your k-means clusters (which were randomly seeded and therefore likely differ from mine) and copy over my precalculated k-means clusters:  
  
```
cd 18_4Apix_box96_z128_b8_v103
rm -r analyze.49/kmeans20
cp -r analyze.49_precalculated/kmeans20 analyze.49
cd ..
```
  
Now, we can use ```filter_star``` to exclude particles from the selected clusters: 
  
```
tomodrgn filter_star \
    18_4Apix_box96_z128_b8_v103/run_optimisation_set_fixedpaths_tomodrgn_preprocessed.star \
    --starfile-type optimisation_set \
    -o 18_4Apix_box96_z128_b8_v103/run_optimisation_set_analyze49_k20_omitjunk.star \
    --labels 18_4Apix_box96_z128_b8_v103/analyze.49/kmeans20/labels.pkl \
    --labels-sel 0 2 3 4 5 12 \
    --action drop
```
  
>:triangular_flag_on_post: How can we validate that the particles we filtered out were junk or other non-ribosomal particles?   
  
>:white_check_mark: You can also filter particles by supplying a ```.pkl``` file containing the indices of particles you want to retain or drop to filter_star. This index file can be generated pythonically and tailored to filter based on any criterion you are interested in. Since we found that many of our junk particles had outlier values of _rlnCoordinateZ, try generating a ```.pkl``` that only include particles indices for particles with _rlnCoordinateZ between 175 and 300. Use that file to filter your particle stack instead, and train a model on that filtered particle stack. Did you do a better or worse job filtering by this approach? 
  
    
[back to top](#um-cryo-et-workshop-processing-heterogeneous-data-with-tomodrgn-and-siren)  
  
  
## Training a final tomoDRGN model  

Now that we have filtered the tomoDRGN model, we can train a final model that we can use to analyze structural heterogeneity within these ribosomes. 
  
```
tomodrgn train_vae \
    18_4Apix_box96_z128_b8_v103/run_optimisation_set_analyze49_k20_omitjunk.star \
    -o 03_4Apix_box96_z128_b8_v103_filtered \
    -n 50 \
    -b 8 \
    --zdim 128 \
    --recon-dose-weight \
    --recon-tilt-weight \
    --l-dose-mask \
    --image-dose-weighted false \
    --image-ctf-premultiplied false \
    --source-software warptools \
    --num-workers 2 \
    --prefetch-factor 2 \
    --pin-memory \
    --datadir /work/data/EMPIAR-10499/warp_tiltseries/particleseries/
```  
  
As before, we will switch to using pre-calculated results for consistency. These results can be found in ```/work/data/EMPIAR-10499/tomodrgn/19_4Apix_box96_z128_b8_v103_filtered```. After copying the precalculated results folder to ```/scratch```, we will start by analyzing the convergence of the trained tomoDRGN model, this time using tomoDRGN's built-in ```convergence_vae``` function.  
  
```
cp -r /work/data/EMPIAR-10499/tomodrgn/19_4Apix_box96_z128_b8_v103_filtered ./
tomodrgn convergence_vae 19_4Apix_box96_z128_b8_v103_filtered/ --epoch 49
```
  
Take a moment to look through the results folder (```19_4Apix_box96_z128_b8_v103_filtered/convergence.49```) by yourself. What do you notice? What plots seem helpful for determining whether the model has converged?  
  
>:triangular_flag_on_post: Let's walk through the results together. What epoch should we select for further analysis?
  
Based on the convergence analysis above, we'll proceed with epoch 49. This time, when we run ```tomodrgn analyze```, we'll generate a large ensemble of volumes. This will help us better identify diverse structural states present in the dataset, and will provide the inputs for the next step of processing, where we aim to identify variable regions in the volume in an automated fashion.  
   
```
tomodrgn analyze 19_4Apix_box96_z128_b8_v103_filtered/ --epoch 49 --ksample 500
``` 
  
Again, for the sake of consistency, we will switch over to precalculated results, located in ```19_4Apix_box96_z128_b8_v103_filtered/analyze.49_precalculated```. Before we move on to automated analysis of this volume ensemble, take a couple of minutes to look through the 500 k-means cluster center volumes yourself in ChimeraX. It's important to develop an intuition for what is changing in your volumes, in addition to using automated and quantitative tools. 
  
  
>:triangular_flag_on_post: What kinds of interesting (or non-interesting) structural heterogeneity are we seeing from this model? How does this map to places we would expect to see heterogeneity, based on either the biology of the ribosome or the regions of low local resolution in the consensus refinement?    
  
  
[back to top](#um-cryo-et-workshop-processing-heterogeneous-data-with-tomodrgn-and-siren)  
  
  
## Identifying variable regions with SIREn   
    
In addition to identifying heterogeneity by manual inspection, it can be useful to identify variable regions that vary across the ensemble in a quantitative and automated fashion. For that purpose, we will use [SIREn](https://www.sciencedirect.com/science/article/pii/S0969212625000577), which uses statistical inference to identify groups of voxels ("structural blocks") that come and go together across the ensemble. Notably, although we apply SIREn specifically to cryo-ET data here, it is agnostic to upstream processing methods and can be equivalently applied to single-particle data.  
   
SIREn takes as input a large ensemble of 3D density maps, with optimal algorithm performance for ensembles of ~500 maps. Each map is binarized, and then the co-variance of voxel occupancy across the ensemble is used to infer structural blocks.  
  
Because SIREn is sensitive to the applied binarization threshold, it is important to binarize correctly. To that end, we implemented a 3D CNN trained on thousands of maps and downloaded from the EMDB and their accompanying annotated contour levels. This 3D CNN can automatically generate an estimated binarization threshold for any density map. Although we will not use this functionality today, it is worth noting that this 3D CNN can be installed as a ChimeraX plug-in and used for routine data processing tasks outside of just SIREn. Instructions on how to install the ChimeraX plug-in can be found [here](https://github.com/mariacarreira/calc_level_ChimeraX).  
  
First, we will make a directory to store our SIREn results in:
  
```
cd 19_4Apix_box96_z128_b8_v103_filtered/analyze.49_precalculated
mkdir siren
```  
  
Now, we will preprocess the k-means 500 cluster center volumes generated by ```tomodrgn analyze``` to prepare them for evaluation by the 3D CNN. 

```
conda deactivate
conda activate siren
siren preprocess --voldir kmeans500/ --outdir siren/
```
  
This should have created a new folder ```tomodrgn/19_4Apix_box96_z128_b8_v103_filtered/analyze.49_precalculated/siren/normalized``` that holds a normalized copy of each map at the original box size (here, 96 pix) and an additional subdirectory ```tomodrgn/19_4Apix_box96_z128_b8_v103_filtered/analyze.49_precalculated/siren/normalized/downsampled``` that stores a downsampled copy of each map. Because the SIREn algorithm is computationally expensive, and scales combinatorially with box size, we typically downsample maps to 64 pix before running SIREn. Therefore, we will evaluate the 3D CNN on these downsampled volumes.   
  
```
siren eval_model \
    --voldir siren/normalized/ \
    --outdir siren/ \
    --weights_file ../../weights_5e6.pth \
    --normalize_csv siren/normalized/map_stats_downsampled.csv 
```  
  
This generates a new file, ```tomodrgn/19_4Apix_box96_z128_b8_v103_filtered/analyze.49_precalculated/siren/predictions.csv```, which has encodes the predicted binarization threshold for every volume in the ensemble. Before we go any further, we should sanity-check that these look reasonable. To do so, open a couple of example volumes from ```tomodrgn/19_4Apix_box96_z128_b8_v103_filtered/analyze.49_precalculated/siren/normalized/downsampled``` in ChimeraX and visualize them at their predicted contour levels. Do these levels seem reasonable? 
  
Finally, we run SIREn itself. This is done in two steps: first, sketching out initial blocks using a random subset of the voxels, and then iteratively expanding these blocks. In the first step, it is important to apply the correct pixel size in angstroms from the downsampled (box 64) volumes, because SIREn applies a locality-scaling factor that determines how strong the statistical evidence for co-occupancy must be based on how far apart two voxels are in real space.   
   
```  
siren sketch_communities \
    --voldir siren/normalized/downsampled/ \
    --outdir siren/ \
    --threads 16 \
    --apix 6 \
    --filter \
    --binfile siren/predictions.csv
     
siren expand_communities \
    --config siren/00_sketch/config.pkl \
    --blockdir siren/00_sketch/ \
    --threads 16 \
    --filter
```
  

>:white_check_mark: Note that we used the ```--filter``` flag above. This flag excludes a small number of volumes that have outlier predicted binarization thresholds. In practice, we find this useful for excluding residual junk, which can significantly challenge the SIREn algorithm in some cases. How do your results compare if you run SIREn without --filter on this volume ensemble?  
  
While your SIREn command is running, examine the pre-calculated results, found in ```19_4Apix_box96_z128_b8_v103_filtered/analyze.49_precalculated/siren_precalculated``` by opening the blocks in ChimeraX. How do the inferred structural blocks compare to regions you manually identified as variable? 

>:triangular_flag_on_post: What structural blocks might represent interesting structural heterogeneity in these ribosomes?   
  
  
[back to top](#um-cryo-et-workshop-processing-heterogeneous-data-with-tomodrgn-and-siren)  
  
  
## Measuring the occupancy of SIREn blocks
  
We'd now like to measure how these blocks are occupied across the structural ensemble. One reason this is useful is because it allows us to identify subsets of particles enriched for a particular feature, which we can export back to RELION for validation via traditional refinement procedures.  
  
Because SIREn blocks are simply binarized 3D volumes of the same dimensions as the input density maps, we can treat them directly as masks for measuring the occupancy of each block in each of the 500 sampled volumes. The code to do that is implemented in the python script found in ```/work/data/EMPIAR-10499/maven/calc_occupancy.py```, which is adapted from [MAVEn](https://github.com/lkinman/maven).    
  
First, we will make a directory to store our results in:
  
```
cd 19_4Apix_box96_z128_b8_v103_filtered/analyze.49_precalculated
mkdir maven
```  
  
The ```calc_occupancy.py``` script expects input masks to follow the specific naming convention "Mask_{name}_chain{X}.mrc", where the curly brackets indicate variables. It is relatively straightforward to rename the SIREn blocks so they follow this naming convention:  
  
```
cd maven
mkdir 00_masks
cp ../siren_precalculated/02_expand/*.mrc 00_masks/
cd 00_masks
for i in *.mrc; do mv $i ${i%.mrc}_chaina.mrc; done
for i in *.mrc; do mv $i Mask_block${i#block_}; done
```  
  
Now we can actually measure occupancy, supplying the directory containing our downsampled maps, our (renamed) masks, and the predicted binarization thresholds.  
  
```
cd ../..
python /work/data/EMPIAR-10499/maven/calc_occupancy.py \
    --mapdir siren_precalculated/normalized/downsampled/ \
    --maskdir maven/00_masks/ \
    --outdir maven/ \
    --binfile siren_precalculated/predictions.csv
```  
  
  
[back to top](#um-cryo-et-workshop-processing-heterogeneous-data-with-tomodrgn-and-siren)  
  
  
## Selecting particles based on SIREn block occupancy  
    
We now want to select k-means clusters that have high occupancy of features of interest that we identified using SIREn. To do that, we will launch the ```tomodrgn_interactive_viz.ipynb``` jupyter notebook created by ```tomodrgn analyze```.  

**ADD INSTRUCTIONS FOR LAUNCHING JUPYTER NOTEBOOK**

Add the following packages to the set of packages that are imported at the top of the notebook. 

```
import glob
import string 
import pickle
```
  
Now run the import cells so we have all the packages we need to do our calculations. Then open a new cell and run the following: 
  
```
occupancies = 'maven/occupancies.csv'
chains = {i.split('_')[-2]: [i.split('_')[-2]] for i in glob.glob('maven/00_masks/*.mrc')}

df = pd.read_csv(occupancies, index_col = 0, header = [0,1]).dropna(axis = 1)
df_rename = pd.DataFrame(index = df.index)
alpha_list = string.ascii_lowercase

for col in df.columns:
    pdb_name, chain = col
    chain_ind = alpha_list.index(chain)
    chain_id = chains[pdb_name][chain_ind]
    
    df_rename[chain_id] = df[col]
```  
  
This creates a dataframe (```df_rename```) that reports the occupancy of every block for each volume. Take a moment to familiarize yourself with this dataframe.  
  
>:triangular_flag_on_post: What does the distribution of block 20 occupancies look like? 

Let's take a look at a couple of volumes with high occupancy of block 20. To identify k-means 500 cluster center volumes with occupancy of block 20 greater than ~50%, open a new cell and run the following code: 
  
```
selected_classes = df_rename[df_rename['block20'] > 25].index
```
  
Open these volumes in ChimeraX. Do these volumes indeed have high occupancy of the indicated block? Do they represent a structural state that is worth validating or investigating more?
  
If we want to investigate or validate this structural state via a traditional 3D refinement, we can generate filtered version of a .star file containing only particles from the selected classes defined above. 

```
labels = pd.Series(utils.load_pkl('kmeans500/labels.pkl'))
selected_particles = labels[labels.isin(selected_classes)].index
with open('model19_analyze49_block20_high.pkl', 'wb') as f:
    pickle.dump(sel_particles, f)
```
  
This saves a ```.pkl``` file with the indices of the selected classes. We can now use the same ```filter_star``` command we used earlier to create a curated stack with only these particles in the command line.  
  
```
tomodrgn filter_star \
    19_4Apix_box96_z128_b8_v103_filtered/run_optimisation_set_analyze49_k20_omitjunk_tomodrgn_preprocessed.star \
    -o ./model19_analyze49_block20_high_optimisation_set.star \
    --starfile-type optimisation_set \
    --ind 19_4Apix_box96_z128_b8_v103_filtered/analyze.49/model19_analyze49_block20_high.pkl \
    --action keep
```
  
This only has 265 particles! Which means further 3D refinement is likely to be challenging. This highlights an important consideration for processing heterogeneous datasets: larger datasets are generally preferable, because larger datasets will allow you to observe and validate increasingly rare structural states. 

Nonetheless, we can try a quick validation using ```tomodrgn backproject_voxel```:
  
```
tomodrgn backproject_voxel \
    model19_analyze49_block20_high_optimisation_set.star \
    --output 04_backproject/backproject_weighted.mrc \
    --recon-dose-weight \
    --recon-tilt-weight \
    --source-software warptools \
    --image-dose-weighted false \
    --image-ctf-premultiplied false
```
  
Open the backprojected volumes (```04_backproject/backproject_weighted.mrc``` and ```04_backproject/backproject_weighted_filt.mrc```) in ChimeraX. Also open the initial backprojections from the full stack (```00_backproject/backproject_weighted.mrc``` and ```00_backproject/backproject_weighted.mrc```). How do the two sets of backprojections compare? Is the density near the peptide exit tunnel a hallucination? Did we successfully enrich for particles bearing this feature?

Now, repeat this analysis with block 0 on your own. 
  
>:triangular_flag_on_post: Are there regions that we might have expected to see a SIREn block where we didn't find one, based on either the consensus map or the volume ensemble?  
  
  
[back to top](#um-cryo-et-workshop-processing-heterogeneous-data-with-tomodrgn-and-siren)  
  
  
## Refining SIREn results with a filtered volume ensemble
    
As mentioned above, SIREn is not very robust to the presence of junk volumes in the ensemble. As we noticed when looking at our k-means 500 volumes manually, there is still junk in this volume ensemble! We attempted to get rid of some of it using ```--filter``` when we ran SIREn, but that was only partly effective.  
  
To try to refine our SIREn results, we can manually filter our volume ensemble. First, let's make a new directory to store our SIREn results in:

```
cd tomodrgn/19_4Apix_box96_z128_b8_v103_filtered/analyze.49_precalculated
mkdir siren_filt
```
  
Now we can copy our kmeans 500 volumes into this new directory. This is useful because we can just delete junk volumes from this directory, and then feed this filtered ensemble directly to ```siren sketch_communities``` as the ```--mapdir```. 

```
cp -r siren_precalculated/normalized/downsampled/ siren_filt/filtered_maps
```  
  
Now open the k-means maps in ChimeraX. Set them all to a reasonable contour level (I recommend 0.01-0.02) and go through them one by one to identify volumes you want to exclude. 
  
Once you've identified the set of volumes you want to exclude, delete those volumes from the ```siren_filt/filtered_maps``` directory using ```rm```. 

You should try to do this on your own, but as an example, here are the volumes I excluded for the precalculated results.  
  
<details>
<summary>Example filtering</summary>

```
cd siren_filt/filtered_maps
for i in 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 024 025 026 027 028 029 030 031 032 033 035 036 037 038 039 040 054 055 056 057 061 065 066 067 074 087 093 110 131 293 294 295 296 297 298 299 306 307 310 319 320 332 333 334 335 342 413 420 421 422 423 462 485 486 487 494 495 497 498 499; do rm vol_${i}.mrc; done
```

</details>
  
Now we can run SIREn pretty straightforwardly on this manually-filtered volume ensemble using the same commands as before. Note that here we are omitting ```--filter```.  

```
siren sketch_communities \
    --voldir siren_filt/filtered_maps/ \
    --outdir siren_filt/ \
    --threads 16 \
    --apix 6 \
    --binfile siren/predictions.csv

siren expand_communities \
    --config siren_filt/00_sketch/config.pkl \
    --blockdir siren_filt/00_sketch/ \
    --threads 16
```
  
Once again, we will switch over to looking at pre-calculated results for consistency, located in ```siren_filt_precalculated```. Open these in ChimeraX and compare them to the previous pre-calculated SIREn blocks.

>:triangular_flag_on_post: How do these SIREn blocks compare to those calculated without manual filtering? Are there any features we pick up here that we failed to pick up previously?  
  
Now, we will again measure the occupancy of each of these blocks and examine the distributions of block occupancies. You should try to do this yourself, but the code I used is below for reference. 

<details>
<summary>Code for measuring occupancies of filtered SIREn blocks</summary>

```
cd tomodrgn/19_4Apix_box96_z128_b8_v103_filtered/analyze.49_precalculated
mkdir maven_filt
cd maven_filt
mkdir 00_masks
cp ../siren_filt_precalculated/02_expand/*.mrc 00_masks/
cd 00_masks
for i in *.mrc; do mv $i ${i%.mrc}_chaina.mrc; done
for i in *.mrc; do mv $i Mask_block${i#block_}; done
cd ../..
python ../../../maven/calc_occupancy.py \
    --mapdir siren_filt_precalculated/filtered_maps/ \
    --maskdir maven_filt/00_masks/ \
    --outdir maven_filt/ \
    --binfile siren/predictions.csv
```

</details>    
    

Examine the results in the jupyter notebook as before. Pick a block of interest to you, and look at volumes that have high/low occupancy of that block. Try to filter the input .star file by these indices and perform a backprojection as before to validate that feature.  
  
>:triangular_flag_on_post: Share the results of this exercise with the group. Did anyone identify any interesting structural states?   
  

<details>
<summary>Code for selecting particles with high block 9 occupancy </summary>

If I wanted to select volumes with high occupancy of block 9, I could run the following in the jupyter notebook:

```
occupancies = 'maven_filt/occupancies.csv'
chains = {i.split('_')[-2]: [i.split('_')[-2]] for i in glob.glob('maven_filt/00_masks/*.mrc')}

df = pd.read_csv(occupancies, index_col = 0, header = [0,1]).dropna(axis = 1)
df_rename = pd.DataFrame(index = df.index)
alpha_list = string.ascii_lowercase

for col in df.columns:
    pdb_name, chain = col
    chain_ind = alpha_list.index(chain)
    chain_id = chains[pdb_name][chain_ind]
    
    df_rename[chain_id] = df[col]

selected_classes = df_rename[df_rename['block9'] > 60].index
labels = pd.Series(utils.load_pkl('kmeans500/labels.pkl'))
selected_particles = labels[labels.isin(selected_classes)].index
with open('model19_analyze49_filt_block9_high.pkl', 'wb') as f:
    pickle.dump(sel_particles, f)
```

and then run the following via the command line:

```
tomodrgn filter_star \
    19_4Apix_box96_z128_b8_v103_filtered/run_optimisation_set_analyze49_k20_omitjunk_tomodrgn_preprocessed.star \
    -o ./model19_analyze49_filt_block9_high_optimisation_set.star \
    --starfile-type optimisation_set \
    --ind 19_4Apix_box96_z128_b8_v103_filtered/analyze.49/model19_analyze49_filt_block9_high.pkl \
    --action keep 

tomodrgn backproject_voxel \
    model19_analyze49_filt_block9_high_optimisation_set.star \
    --output 05_backproject/backproject_weighted.mrc \
    --recon-dose-weight \
    --recon-tilt-weight \
    --source-software warptools \
    --image-dose-weighted false \
    --image-ctf-premultiplied false
```

</details>    


## References

1. Tegunov, D., and Cramer, P. (2019). Real-time cryo-electron microscopy data preprocessing with Warp. Nat Methods *16*, 1146–1152. doi: [10.1038/s41592-019-0580-y](https://doi.org/10.1038/s41592-019-0580-y).
2. Tegunov, D., Xue, L., Dienemann, C., Cramer, P., and Mahamid, J. (2021). Multi-particle cryo-EM refinement with M visualizes ribosome-antibiotic complex at 3.5 Å in cells. Nat Methods *18*, 186–193. doi:[10.1038/s41592-020-01054-7](https://doi.org/10.1038/s41592-020-01054-7).
3. Chaillet, M.L., van Loenhout, J., Leung, M.R., Burt, A., and Tegunov, D. (2026). MissAlignment teaches itself better cryo-ET tilt-series alignment by making it worse. bioRxiv 2026.04.29.721716. doi: [10.64898/2026.04.29.721716](https://doi.org/10.64898/2026.04.29.721716)
4. Burt, A., Toader, B., Warshamanage, R., von Kügelgen, A., Pyle, E., Zivanov, J., Kimanius, D., Bharat, T.A.M. and Scheres, S.H.W. (2024). An image processing pipeline for electron cryo-tomography in RELION-5. FEBS Open Bio *14*, 1788-1804. doi: [10.1002/2211-5463.13873](https://doi.org/10.1002/2211-5463.13873).
5. Powell, B.M., and Davis, J.H. (2024). Learning structural heterogeneity from cryo-electron sub-tomograms with tomoDRGN. Nat Methods *21*, 1525–1536. doi: [10.1038/s41592-024-02210-z](https://doi.org/10.1038/s41592-024-02210-z).
6. Kinman, L.F., Carreira, M.V., Powell, B.M., and Davis, J.H. (2025). Automated model-free analysis of cryo-EM volume ensembles with SIREn. Structure *33*, 974-987.e4. doi: [10.1016/j.str.2025.02.004](https://doi.org/10.1016/j.str.2025.02.004). 
7. Kinman, L.F., Powell, B.M., Zhong, E.D., Berger, B., and Davis, J.H. (2023). Uncovering structural ensembles from single-particle cryo-EM data using cryoDRGN. Nat Protoc *18*, 319–339. doi: [10.1038/s41596-022-00763-x](https://doi.org/10.1038/s41596-022-00763-x).
8. Sun, J., Kinman, L.F., Jahagirdar, D., Ortega, J., and Davis, J.H. (2023). KsgA facilitates ribosomal small subunit maturation by proofreading a key structural lesion. Nat Struct Mol Biol *30*, 1468–1480. doi: [10.1038/s41594-023-01078-5](https://doi.org/10.1038/s41594-023-01078-5).


## Appendix 1: Custom Warp build 
  
First, navigate to a desired directory and git clone the Warp repository: 
  
```
cd /path/to/dir
git clone https://github.com/warpem/warp/ 
```  
  
Open ```warp/WarpLib/TiltSeries/TiltSeries.ReconstructParticleSeries.cs``` in your editor of choice (e.g. Visual Studio Code) and *manually delete the following two lines*.  
  
```
ImagesFT.Multiply(CTFs);
ImagesFT.Multiply(RelionWeights)
```  
  
Now run the following from inside your local Warp repo.  
  
```
conda env create -f warp_build.yml
conda activate warp_build
./scripts/build-native-unix.sh
./scripts/publish-unix.sh
```  
  
Anytime you want to use this custom build of Warp, edit the following according to your local paths, and then run:  
  
```
conda activate warp_build
export PATH="/path/to/dir/warp/Release/linux-x64/publish:$PATH"
export LD_LIBRARY_PATH="/path/to/dir/warp/Release/linux-x64/publish:/path/to/conda/envs/warp_build/lib:$LD_LIBRARY_PATH"
```  
  
As described below, *most Warp preprocessing should still be done with the standard Warp build*. Only the final particle export before tomoDRGN training should be done with the custom Warp build.  
  
  
## Apppendix 2: Particle preprocessing pipeline 
  
The following is a complete description of how the downloaded frame and tiltseries data from [EMPIAR-10499](https://www.ebi.ac.uk/empiar/EMPIAR-10499/) was processed to generate the particle stacks used in today's workshop. These instructions largely (but not perfectly) follow [the WarpTools documentation](https://warpem.github.io/warp/user_guide/warptools/quick_start_warptools_tilt_series/), with the notable exception that [Miss-Alignment](https://www.biorxiv.org/content/10.64898/2026.04.29.721716v1) was used to refine the tilt series alignments. 
  
First, we created settings files for the frame and tilt series data.  
  
```
WarpTools create_settings --folder_data frames --folder_processing warp_frameseries --output warp_frameseries.settings --extension "*.tif" --angpix 1.7005 --gain_path CountRef_TS_01_001_-0.0.dm4 --gain_flip_y --exposure 3.25  
  
WarpTools create_settings --output warp_tiltseries.settings --folder_processing warp_tiltseries --folder_data tomostar --extension "*.tomostar" --angpix 1.7005 --gain_path CountRef_TS_01_001_-0.0.dm4 --exposure 3.25 --tomo_dimensions 4000x4000x1000
```  
  
Then each frame series was motion-corrected and CTF estimated.  
  
```
WarpTools fs_motion_and_ctf --settings warp_frameseries.settings --m_grid 1x1x12 --c_grid 2x2x1 --c_range_max 7 --c_defocus_max 8 --c_use_sum --out_averages --out_average_halves --perdevice 2 --device_list 0 1
```    
  
Tilt series were imported:  
  
```
WarpTools ts_import --mdocs mdocs --frameseries warp_frameseries --tilt_exposure 3.24 --min_intensity 0.3 --dont_invert --output tomostar
```  
  
and an initial alignment was performed with [etomo](https://bio3d.colorado.edu/imod/doc/etomoTutorial.html). Because we are going to refine this alignment with Miss-Alignment, we are not too worried about optimizing the alignment in this step. We also checked the defocus hand and then flipped it, since the resulting correlation was negative. 
  
```
WarpTools ts_etomo_patches --settings warp_tiltseries.settings --angpix 10 --patch_size 1000 --initial_axis -85.6 --do_axis_search --device_list 0 1 --perdevice 2  
  
WarpTools ts_defocus_hand --settings warp_tiltseries.settings --check  
  
WarpTools ts_defocus_hand --settings warp_tiltseries.settings --set_flip
```
  
We then prepared the data for Miss-Alignment using ```ts_autolevel```. We also updated the .xml files in Warp as described in the [Miss-Alignment GitHub docs](https://github.com/warpem/miss-alignment/blob/main/docs/usage.md).  
  
```
WarpTools ts_autolevel --settings warp_tiltseries.settings --angpix 10 --patch_size 1000 --device_list 0 1 --perdevice 2
  
python update_warp_xml.py
```  
  
We refined Miss-Alignment using the following command within a Slurm batch submission script:  
  
```  
miss-alignment \
    --config-file /data/laurel/test_system/test_cryoET/EMPIAR_10499/config.yaml \
    --training-devices 0 \
    --reconstruction-devices 1,1,1,1 \
    --dataloaders-per-trainer 5 \
    --start-at-iteration 0 \
    --prepare-stacks 10.0
```  
   
We then returned to Warp to evaluate the CTF across the whole tilt series and finally reconstruct tomograms. Notably, these reconstructed tomograms looked much better than tomograms reconstructed using the initial etomo alignments. 
  
```
WarpTools ts_ctf --settings warp_tiltseries.settings --range_high 7 --defocus_max 8 --device_list 0 1 --perdevice 2  
  
WarpTools ts_reconstruct --settings warp_tiltseries.settings --angpix 10 --device_list 0 1 --perdevice 2
```  
  
We then used [EMD-11999](https://www.ebi.ac.uk/emdb/EMD-11999) to template-match for ribosomes in the reconstructed tomograms, and threshold and extracted selected images.   
  
```
WarpTools threshold_picks --settings warp_tiltseries.settings --in_suffix 11999 --out_suffix clean --minimum 1.8 
  
WarpTools ts_export_particles --settings warp_tiltseries.settings --input_directory warp_tiltseries/matching --input_pattern "*11999_clean.star" --normalized_coords --output_star relion/matching.star --output_angpix 4 --box 96 --diameter 300 --relative_output_paths --2d --perdevice 2 --device_list 0 1
```  
  
These particles were imported into RELION-5, and subjected to a single round of 3D classification and subsequent 3D refinement of the selected classes.   
  
```
`which relion_refine_mpi` --o Class3D/job007/run --ios matching_optimisation_set.star --ref ../warp_tiltseries/template/emd_11999.mrc --firstiter_cc --trust_ref_size --ini_high 60 --dont_combine_weights_via_disc --scratch_dir /ssd/relion_cache/laurel/job007 --pool 30 --pad 2  --ctf --iter 25 --tau2_fudge 4 --particle_diameter 300 --K 5 --flatten_solvent --zero_mask --blush  --blush_model v1.0 --oversampling 1 --healpix_order 2 --offset_range 5 --offset_step 2 --sym C1 --norm --scale  --j 9 --gpu ""  --pipeline_control Class3D/job007/  
  
`which relion_refine_mpi` --o Refine3D/job009/run --auto_refine --split_random_halves --i Select/job008/particles.star --tomograms matching_tomograms.star --ref ../warp_tiltseries/template/emd_11999.mrc --firstiter_cc --trust_ref_size --ini_high 60 --blush  --blush_model v1.0 --dont_combine_weights_via_disc --scratch_dir /ssd/relion_cache/laurel --pool 30 --pad 2  --ctf --particle_diameter 300 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale  --j 9 --gpu ""  --pipeline_control Refine3D/job009/ 
```
  
Finally, the particles were re-exported with the custom Warp build.  
  
```
conda activate warp_build  
  
export PATH="/home/lkinman/software/warp/Release/linux-x64/publish:$PATH"  
  
export LD_LIBRARY_PATH="/home/lkinman/software/warp/Release/linux-x64/publish:/home/lkinman/.conda/envs/warp_build/lib:$LD_LIBRARY_PATH"  
  
WarpTools ts_export_particles --settings warp_tiltseries.settings --input_directory warp_tiltseries/matching --input_pattern "*11999_clean.star" --normalized_coords --output_star relion/matching_uncorrected.star --output_angpix 4 --box 96 --diameter 300 --relative_output_paths --2d --perdevice 2 --device_list 0 1
```

And now the particles are ready for use with tomoDRGN v1.0.3+. 





