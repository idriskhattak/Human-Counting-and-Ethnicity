# Human Counting and Ethnicity Detection

Final-year project (BS Artificial Intelligence, Hazara University Mansehra, 2024).

Counts people crossing a line in a video and, in a second stage, runs a demographic
classifier on each detected face. The counting works. **I do not think the second
stage should be deployed** — see [A note on the ethnicity stage](#a-note-on-the-ethnicity-stage).

📄 Full write-up: [idriskhattak.github.io/Idris_Portfolio/projects/people-counting/](https://idriskhattak.github.io/Idris_Portfolio/projects/people-counting/)

## What it does

A frame-by-frame detector alone cannot count people: it tells you someone is present,
not whether they are the same person as the previous frame. Without identity, one
person standing still is counted once per frame.

The pipeline:

1. **Read** — OpenCV pulls frames plus the source width, height and frame rate.
2. **Detect and track** — `model.track(im0, persist=True)` with YOLOv8n. The `persist`
   flag carries track IDs between calls; without it every frame starts fresh.
3. **Count** — a four-point region spanning the frame tallies entries and exits
   separately as tracked IDs cross it.
4. **Write** — annotated frames, with boxes, IDs and trails, go to an output video.

The counting region is a shallow band, not a line:

```python
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
```

A band gives the tracker a few frames to register the crossing. A one-pixel line can
be stepped over between frames by anyone moving quickly, and the crossing is missed.

## Repository contents

| File | Purpose |
| --- | --- |
| `Main.ipynb` | The counting pipeline — detection, tracking, line crossing, video output |
| `object_counter.py` | Ultralytics' `ObjectCounter`, vendored so the region logic can be inspected |
| `Ethinicity_Detection.ipynb` | The demographic stage (see the note below) |
| `yolov8n.pt` | Pretrained COCO weights for the detector |
| `video.mp4`, `IMG_*.{JPG,jpg}` | Sample input |

## Running it

```bash
pip install ultralytics opencv-python shapely
jupyter notebook Main.ipynb
```

Point `cv2.VideoCapture` at your own file and adjust `region_points` to match where
the counting line should sit in your frame. The coordinates in the notebook are tuned
to the included 1080-wide sample.

## Results

| | |
| --- | --- |
| Detector | YOLOv8n, pretrained COCO weights |
| Tracking | Persistent IDs across frames |
| Counting region | Four-point band, 1060px wide, 44px deep |
| Input | Recorded 1080-wide video, processed offline |
| **Counting accuracy** | **Not measured** |

I never hand-labelled a clip to measure counting accuracy against. I watched the
annotated output and the numbers looked right, which is not the same thing. Treat
this as a working prototype, not a benchmarked result.

## A note on the ethnicity stage

The brief's second requirement was demographic estimation per detected person. It is
implemented in `Ethinicity_Detection.ipynb` using DeepFace's race classifier, and I
would not deploy it.

The call runs with `enforce_detection=False`. That flag exists so the pipeline does
not crash when no clear face is found — but it means the classifier returns a
confident label regardless. A blurred face, the back of a head, someone at the edge
of the frame: all get a label, all get a confidence score, none of it grounded in a
face the model actually resolved.

Underneath that, the model predicts a categorical race label from appearance. Its
accuracy is not evenly distributed across the groups it claims to distinguish, its
categories are whichever ones its training set happened to use, and there is no
threshold at which a wrong label here is harmless. Counting footfall is a reasonable
thing to automate. Sorting people into racial categories from camera footage is not
the same kind of task, and treating it as one more model call is the error.

The code is here because it was the assignment. Given the choice now I would propose
a different second stage — dwell time, queue length, or occupancy against a capacity
limit — all of which answer the underlying operational question without classifying
anybody.

## What I would do differently

- Hand-label one minute of footage and measure counting accuracy against it. Without
  that number this is a demo, not a result.
- Drop the demographic stage.
- Test the counting band against people moving at different speeds. Those coordinates
  were tuned by watching one video until the numbers looked right, so they are fitted
  to that video and I do not know how they behave on another.
