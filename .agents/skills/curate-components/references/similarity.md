# Triage `tsf model similar` clusters

The miner is static (`src/tsflab/catalog/similarity.py`): it parses model and
component code, alpha-renames identifiers, keeps library calls (`torch.fft.rfft`,
`nn.Linear`, `F.gelu`) and tensor methods (`.permute`, `.softmax`), and scores
pairs by 0.7 x shingle Jaccard + 0.3 x library-call Jaccard (exact copies score 1).
Glue (fewer than two library functions) and paired `__init__` declarations are
dropped. A score is a nomination, never evidence of equivalence.

For each cluster, in rank order:

1. Open every member at its `location`. Write down the operator in one line of
   mathematics (inputs, axes, output) for each member.
2. Classify:
   - **reuse-existing** - a component member exists and the model copy matches
     its equations, shapes, normalization, masking, initialization, state and
     outputs: migrate the model to the component.
   - **extract-new** - two or more models share the same operator and it is
     paper-neutral (a transform, an attention variant, a recurrent cell, a
     calendar or position encoding): extract one component.
   - **material variant** - same shape, different mathematics (e.g. a mask, a
     residual order, an activation): keep local, and name the difference in each
     card's Differences so the next pass does not re-nominate it blindly.
   - **coincidence** - shared boilerplate with no common operator: ignore.
3. Record the decision, the members, and why, in the pass report; extraction
   then follows the skill steps (fixtures before editing, equivalence check,
   card with role, slot, fits, and interface).

When an extracted component consolidates code from several papers, its card's
`origin` names the first paper that introduced the operator and `origin_models`
lists the consumers it was extracted from, so the component is traceable to the
literature (consumers themselves are derived from imports).
