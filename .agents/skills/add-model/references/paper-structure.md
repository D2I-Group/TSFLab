# Extract paper structure

Produce an implementation map precise enough that another worker can implement or
audit the model without guessing a defining operation.

Sources: the primary paper and supplement; the official repository pinned to a
revision with its license recorded; the current component catalog
(`uv run tsf catalog list --kind component`, one L0 line each).

1. Keep paper facts, official-code clarifications, and local design choices in
   separate fields. Do not copy source text or code into the map.
2. Record, in execution order:
   - task, inputs, covariates, tensor axes, output, and probabilistic semantics;
   - preprocessing, normalization, decomposition, embeddings, main blocks, head,
     and inverse transforms;
   - defining equations with section or equation references;
   - loss, auxiliary objectives, initialization, defaults, train/eval differences;
   - shape invariants, sequence constraints, marks or adjacency contracts, edge cases;
   - official-code clarifications with revision and source path;
   - unspecified details that would materially change fidelity.
3. Give every implementable operation a component decision: `reuse-existing`
   (component and matched contract), `extract-new` (the standalone contract it would
   expose and its expected consumers), or `model-local` (why it is paper-specific).

   ```bash
   uv run tsf catalog search --kind component <operation-and-contract-terms>   # L0 lines
   uv run tsf catalog show <candidate...>                     # L1 of every candidate; --depth 2 only if needed
   ```

   Compare mathematics, axes, normalization, masking, residual order,
   initialization, state, and outputs; a similar name is not a match.
4. Mark paper-neutral operators (transforms, attention variants, routers,
   normalizations) as `extract-new` so they become reusable building blocks.

Success: every defining operation has a decision, facts and inferences are labeled,
the official revision is pinned. Stop and surface an ambiguity when choosing silently
would change the method.
