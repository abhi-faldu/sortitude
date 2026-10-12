# Live samples

Real frames saved by the Coral Dev Board during our live webcam demo. The label and
confidence in the top-left corner were drawn by the board itself at capture time.

- `correct/` - frames the model got right (plastic bottle, watch -> metal, boxes -> cardboard,
  brochure -> paper, candy wrapper -> trash).
- `misclassified/` - typical failures. The model is trained on TrashNet's clean, single-object
  photos on white backgrounds, so in a real room it is overconfident and has no way to say
  "nothing here" (an empty wall becomes *paper*). See [`docs/IMPROVEMENT_PLAN.md`](../docs/IMPROVEMENT_PLAN.md).
- `test_sample.jpg` - input image used by `board/benchmark.py`.

The full capture session (~150 frames) is not published because some frames show people.
