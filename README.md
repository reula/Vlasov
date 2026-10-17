# Vlasov
 Vlasov equation solver

We solve the equations:
```math
\begin{align}
\partial_t f &+ v \cdot \partial_x f + q/m E \cdot \partial_v f = 0 \\
\partial_x \cdot E &= \rho \\
\partial_t E &= - S
\end{align}
```

Here we do it in one dimension.

## Repository hygiene

Run data (`Results/`, `*.jld2`), reference books/papers (`*.pdf`) and generated
figures/animations (`E-V/Images/`, `*_fps5.gif`) are intentionally **not** tracked.
Keep them outside the repository (or in the ignored `Results/` directory).

Notebook outputs are stripped automatically when committing. Configure the
filter once per clone:

```bash
git config filter.nbclean.clean 'jupyter nbconvert --to notebook --stdin --stdout --ClearOutputPreprocessor.enabled=True'
git config filter.nbclean.smudge cat
git config filter.nbclean.required true
```



