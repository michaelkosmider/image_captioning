Welcome to my image captioning project! I implemented and trained a vision transformer from scratch. This readme will be completed soon.

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



