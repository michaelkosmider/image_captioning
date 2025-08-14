Welcome to my image captioning project! I made this project to improve my understanding of the MAE paper, image captioning, and also for fun. In this writeup I will go into detail about the current methodology and implementation used to train a ViT + decoder model on the MS COCO dataset, achieving clear and structured descriptions given images as input. Everything here is built from scratch, and I did not use any pre-trained models. If you like seeing nitty gritty implementation details, then this writeup is for you. 

In a nutshell, the model is a caption decoder hooked up to a ViT. The ViT is first trained on the COCO images only, while being part of a masked autoencoder. This allows it to learn complex features from the images. Next, the full captioner is trained on images + captions, where the captions are tokenized using byte pair encoding and a vocabulary size of 5000. To avoid catastrophic forgetting, the encoder part of the captioner uses small learning rates and is slowly unfrozen during training.

It is important to follow the source code alongside this writeup. I recommend opening it up in another tab and viewing them side by side. There are also numbered examples in code_snippets.ipynb which I shall refer to directly.

## The Dataset

The 2017 MS COCO captioning dataset consists of 118K training and 5K validation images. Each image is equipped with 5 human written captions, typically 20 words long. Furthermore, the dataset contains 123K unlabeled images and 41K test images. I'll now describe the code in process_coco.py. This script is run only once, and it is the first script that must be executed.

In section 1, you will find code that downloads the train, val, test, and unlabeled images, each into their own folder. It also downloads the annotations as two separate .json files (training and validation). 

Note: I abstracted away all file locations in a dict named "paths", which you can find in paths.py. 

In section 2, I instantiate a Hugging Face tokenizer which uses byte pair encodings, and then train it on the training captions only. This tokenizer removes accents and capitalization in order to reduce the necessary vocabulary size. The object "train_captions" on line 106 is a list of strings (captions), and Example 1 should clarify the structure of the annotations file. 

Finally, section three tokenizes every caption and creates several data structures. Have a look at examples 2-4 in code_snippets.ipynb to understand what is contained in these structures. Example 2 also illustrates how load and display an image, and how to use the tokenizer to decode.

## The Datasets and DataLoaders.

The Dataset definitions are found in coco_loader.py. Their implementations, and in particular the __getitem__ methods, depend heavily on the data structures generated in process_coco.py. You'll find three distinct Dataset classes: CaptFirstDataset, ImgFirstDataset, and ImageOnlyDataset. Have a look at the docstrings for details.

Each Dataset serves a different purpose in the training and evaluation pipeline.

- CaptFirstDataset: The captioner training loop needs it to iterate over over all (image, caption) pairs. Note that each image appears in 5 different pairs as there are 5 captions per image. 

- ImageFirstDataset: The BLEU score evaluation loop needs it to iterate over all (image, [5 captions], image_id) triplets. 

- ImageOnlyDataset: The MAE training loop needs it to iterate over all images.

The corresponding data loaders are returned by get_coco_loader(), whose args specify the desired type of dataloader. Examples 5 through 7 demonstrate usage of get_coco_loader().

## The Models.

The model implementation is found inside image_captioner.py, and the hyperparameters are found inside the *_config.yaml files. Example 8 shows how to initialize the masked auto encoder, as well as the full captioner model, as specified by the configs. 

In the file you will find four classes, all of which are built on top of the transformer_components package (which I also wrote from scratch, see Transformer Tutorial with PyTorch): ImageEncoder, ImageDecoder, ImageAutoEncoder, and CaptionDecoder. Let's look at each one.

The ImageEncoder precisely the ViT used for the captioning task. It contains a PatchEmbedding Module, as well as a TransformerEncoder which comes from transformer_components. The PatchEmbedding Module projects each image patch to an embedding, thus converting the image into a sequence of patches going from left to right, top down. Each embedding in the sequence is then injected with a positional encoding. The sequence is fed through the TransformerEncoder, which is an exact implementation of the encoder from "Attention is all you Need". 

The ImageDecoder takes in the output from the ViT, which consists of unmasked patch embeddings, and attempts to predict the pixel values of the masked patches. The ImageDecoder is discarded and only the ViT is retrieved after training the autoencoder. 

The ImageAutoEncoder is a convenience class which combines the ImageEncoder and ImageDecoder. 

Finally, the CaptionDecoder builds upon the TransformerDecoder, which is an exact implementation of the decoder from the paper. Specifically, it embeds tokens and adds positional encodings, and also projects the last layer's output features to vocabulary size. Additionally, CaptionDecoder accepts KV cache with a position, making it compatible with the generate method of the TransformerEncoderDecoder from transformer_components. The full captioning model is a TransformerEncoderDecoder, where the encoder is an ImageEncoder, and the decoder is a CaptionDecoder.  

## Encoder Training.

In train_encoder.py you will find the main logic of the training loop. You can run it with 

"python train_encoder.py" 

to train from scratch, or with 

"python train_encoder.py -c checkpoint*.pt" 

to train from a checkpoint. Checkpoints are automatically saved at the end of each epoch, with * representing the number of epochs completed.

Section 1 of train_encoder.py reads in the hyperparameters and training configuration, sets the device, and optionally loads in a checkpoint. 

Section 2 instantiates all of the training loop elements, possibly from a checkpoint: the masked auto encoder itself, training history, AdamW optimizer, warmup + cosine annealing scheduler, gradient scaler (helpful for mixed precision training), MSE loss function, and the ImageOnlyDataset loader. 

Section 3 is the training loop, and line 196 specifically shows you how to use the autoencoder. You have to pass in a batch of images, as well as a corresponding batch of unmasked positions which must have shape (N_batch, # of masked patches). Thus, for each image of the batch, you must randomly select NUM_PATCHES - NUM_MASKED_PATCHES patches to remain unmasked. This selection is reflected in the tensor unmasked_position. The model then reconstructs ALL image patches on line 196, including the unmasked ones. 

Computing the loss is a little bit tricky, because it is only computed on the predictions for masked patches. We don't care whether the model can predict unmasked patches since it just needs to learn the identity function for that. Again, the autoencoder predicts all patches, but the function torch.gather allows us to select the predicitions for masked patches only, as well as the masked patches themselves. Below is a visualization of the entire input to loss pipeline, clarifying what torch.gather actually does. Hopefully it also clarifies the role of unmasked_patches in the interaction between the encoder and decoder.

## Captioner Training. 

Coming soon!

## Incomplete Notes to be incorporated.

methods and implementations

Model 

- Built my own transformer implementation from scratch, including beam search with KV caching.
- The ImageAutoEncoder consists of ImageEncoder + ImageDecoder, whereas the captioner model is an ImageEncoder + CaptionDecoder. I implemented all of these from scratch too, built on top of my transformer implementation. Found in image_captioner.py

Training
- AdamW with warmup and cosine annealing
- mixed precision training with gradient scaling
- unfreezing + lower LR groups and label smoothing for captioner

Data and Augmentations

- Download and tokenization in process_coco.py.
- tokenization using BPE with 5000 word vocab size.
- saving 4 datastructures for the dataset classes to use.
- Captions are truncated after 60 because most are 20 long and rare cases go to 80.
- Created three custom dataloaders from scratch:
    - caption first for training captioner
    - image first for metrics on captioner
    - image only for MAE
- custom collate fns for image_first and caption_first 
- These cocoloaders are acquired with get_coco_loader, exposed inside coco_loader.py
- Originally used rrc + flip + jitter with large encoder model. Now attempting rrc + flip + ra with slightly smaller model to see if overfitting improves.


MAE

- training logic in train_encoder.py
- Patch embedding is a conv2d. 
- MAE to pretrain the image encoder.
    - After patches are embedded, 25% of them are randomly selected (greatly simplified by torch.gather) to go into the encoder.
    - The positions of those selected patches (also called the unmasked patches) are also passed into the encoder, so that the correct positional embeddings are applied.
    - The outputs of the encoder are placed into the decoder input according to their original positions The remaining tokens are all the same "masked token".
    - The decoder attempts to predict pixel values for the masked positions, and the ground truths are the masked patches themselves. 

    changes from 1st attempt to second: doubling the data with unlabeled, lowered dropout to 0, decreased encoder size from 14 to 12, increased lr to 1e-4, but batch size is 64 due to compute (on mac), removed jitter augmentation, broadened scale (maybe not even enough), changed interpolation to 3

Captioner

- training logic in train_captioner.py

Results

- decode_predictions
- bleu score from hugging face

