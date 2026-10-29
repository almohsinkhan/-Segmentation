# Task 1 --- Egg Segmentation Mask Generation

## 1. Objective

The goal of Task 1 is to generate a binary segmentation mask for each
egg in the dataset.

A binary mask uses: - **White (255):** pixels belonging to an egg. -
**Black (0):** background pixels.

The dataset is a Roboflow export resized to **640 × 640**. It contains
approximately 110 images, split into training, validation, and test
folders. The dataset did not provide the pixel-level segmentation masks
needed for this task, so we explored ways to generate them.

> **Important:** The masks produced by this pipeline are model-generated
> pseudo-labels. They should be reviewed before being treated as ground
> truth.

## 2. Dataset and Project Structure

The relevant input folders are expected to be arranged like this:

``` text
Egg/
├── data/
│   ├── train/egg/
│   ├── valid/egg/
│   └── test/egg/
├── models/
│   └── sam_vit_b_01ec64.pth
├── egg_pipeline.py
└── outputs/
```

The exact image counts may vary depending on the exported files. The
initial dataset inspection found about 75 training images, 22 validation
images, and 11 test images, plus two unlabeled images in the training
export.

## 3. Experiments and Decisions

### Experiment 1 --- SAM Automatic Mask Generator

**Approach:** We ran Meta's Segment Anything Model (SAM), using the
ViT-B checkpoint, to automatically propose regions in each image.

**What happened:** - Running on the GPU exceeded the available 4 GB of
VRAM. - Running on the CPU worked, but produced many masks, including
irrelevant regions, overlapping regions, and partial eggs. - One test
image produced 29 candidate masks and took about 79 seconds on CPU.

**Decision:** We did not choose automatic mask generation as the final
method. It did not reliably identify which proposed regions were eggs,
and its output needed too much manual filtering.

### Experiment 2 --- Classical Computer Vision

We tried several traditional image-processing techniques to see whether
the eggs could be separated from the background without a deep-learning
detector.

#### Otsu thresholding and connected components

**Observation:** Bright backgrounds and eggs often had similar pixel
intensities, so thresholding could merge the egg and background into one
region.

**Decision:** Not reliable enough for this dataset.

#### Local brightness maxima

**Observation:** Candidate points appeared on the background as well as
on eggs, so the method did not reliably identify egg locations.

**Decision:** Not suitable as the main detection method.

#### Canny edges and contours

**Observation:** The images produced many unwanted edges and contours.
Background texture and weak boundaries made it difficult to select the
correct egg outlines.

**Decision:** Not reliable enough for automatic mask generation.

**Overall conclusion:** Classical techniques were useful as quick
baselines, but they were too sensitive to image appearance and could not
robustly distinguish eggs from the background.

### Experiment 3 --- Grounding DINO + SAM

This became the selected approach because the two models perform
complementary tasks.

**Grounding DINO** detects objects from a text prompt. We used the
prompt `egg.` to obtain candidate bounding boxes.

**SAM** then takes each detected box as a prompt and predicts a
pixel-level mask for the egg inside that box.

#### Configuration used during testing

-   Grounding DINO model: `IDEA-Research/grounding-dino-tiny`
-   SAM model: ViT-B
-   SAM checkpoint: `models/sam_vit_b_01ec64.pth`
-   Text prompt: `egg.`
-   Detection threshold: `0.25`
-   Text threshold: `0.25`
-   Device: CPU

#### Observed results

-   Grounding DINO took approximately 5--6 seconds per 640 × 640 image
    in the observed CPU tests.
-   SAM mask prediction took approximately 0.06--0.12 seconds for the
    detected objects in those tests.
-   The first test image produced three detections in an earlier
    notebook experiment. The pipeline smoke test later reported one
    detection for that image, two for the second image, and one for the
    third image. These are separate runs/configurations and should not
    be interpreted as a consistent detection count.
-   Visual inspection of five test images during the notebook
    experiments showed promising egg boundaries using the
    DINO-box-to-SAM workflow.

**Decision:** We selected Grounding DINO + SAM because it first locates
eggs and then segments each detected egg. This is more targeted than
asking SAM to propose every possible region in an image.

**Hardware decision:** CPU inference was chosen as the dependable
default because SAM on the available 4 GB GPU exceeded VRAM. It is
slower than a suitable GPU but avoids the memory failure encountered
during the experiment.

## 4. Current Pipeline

The reusable script `egg_pipeline.py` automates the selected workflow:

1.  Find images in the input directory.
2.  Load Grounding DINO and SAM.
3.  Run Grounding DINO with the text prompt `egg.`.
4.  For each detected bounding box, ask SAM for a segmentation mask.
5.  Save individual masks, combined masks, visual overlays, and a CSV
    summary.

Example: smoke-test three test images

``` bash
python egg_pipeline.py \
    --input-dir data/test/egg \
    --output-dir outputs/test \
    --device cpu \
    --max-images 3
```

Example: process the dataset

``` bash
python egg_pipeline.py \
    --input-dir data \
    --output-dir outputs/egg_segmentation \
    --device cpu
```

Run these commands from the project root, where `egg_pipeline.py` is
located, and use the Python virtual environment in which the model
dependencies are installed.

## 5. Output Files

The pipeline writes its results under the selected output directory:

``` text
outputs/egg_segmentation/
├── masks/       # Individual binary mask for each detected egg
├── combined/    # One combined binary mask per image
├── overlays/    # Images with predicted masks, boxes, and confidence scores
└── results.csv  # Per-image status, detection count, timings, scores, and errors
```

-   **Individual masks:** useful when each egg needs its own
    segmentation.
-   **Combined masks:** useful when the task only needs all egg pixels
    grouped into one foreground mask.
-   **Overlays:** useful for visually checking whether the predictions
    match the egg boundaries.
-   **CSV:** useful for checking failures, missing detections, timings,
    and confidence values.

The CSV and image outputs must be inspected after the full run. A
successful run only means the script completed; it does not prove every
egg was found or segmented correctly.

## 6. Limitations

1.  **Qualitative evaluation so far:** the selected method was visually
    inspected on five images, but no ground-truth IoU or Dice score has
    been calculated yet.
2.  **Pseudo-label quality:** model-generated masks may include
    background pixels or miss parts of an egg. Review overlays and masks
    before using them as reference labels.
3.  **Detection misses:** if Grounding DINO does not detect an egg, SAM
    will not generate a mask for that egg.
4.  **Threshold sensitivity:** the detection and text thresholds may
    affect the number of detected eggs. Any change should be tested
    against representative images.
5.  **CPU runtime:** DINO takes several seconds per image in the
    observed tests, so processing the whole dataset may take time.
6.  **Dataset variation:** overlapping eggs, partial eggs, poor
    contrast, and unclear boundaries can make segmentation harder.

## 7. Next Steps

1.  Let the full pipeline finish and check `results.csv` for errors and
    unexpected detection counts.
2.  Review overlays and individual masks across training, validation,
    and test images---not just the first few images.
3.  Correct poor masks manually or create a small manually verified
    ground-truth subset.
4.  If quantitative evaluation is required, compare predictions with
    verified masks using **Intersection over Union (IoU)** and **Dice
    coefficient**.
5.  Once the mask quality is acceptable, keep the selected method and
    move to the next task rather than continuing unnecessary model
    experiments.

## 8. Final Decision Summary

  -----------------------------------------------------------------------
  Method                  Main observation        Decision
  ----------------------- ----------------------- -----------------------
  SAM automatic mask      Many irrelevant,        Rejected as final
  generation              overlapping, or partial approach
                          regions; GPU run        
                          exceeded VRAM           

  Otsu thresholding +     Egg and bright          Rejected
  connected components    background could merge  

  Local brightness maxima Background points were  Rejected
                          also detected           

  Canny edges + contours  Too many irrelevant     Rejected
                          edges and contours      

  Grounding DINO + SAM    DINO localizes eggs;    Selected for the
                          SAM creates a mask from current pipeline
                          each box; promising     
                          results on five         
                          visually inspected      
                          images                  
  -----------------------------------------------------------------------

**Conclusion:** Grounding DINO + SAM is the current best-performing
approach among the experiments tried. It was selected because it
combines text-guided egg detection with prompt-based segmentation. The
result is promising but still requires full-output review and, if the
task requires measured accuracy, comparison against manually verified
masks.
