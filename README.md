# Ten Dots: a physics-informed neural network from first principles

A damped spring is measured ten times in its first 0.3 seconds, with noise. What does it do for the next 0.7?

This repository trains two identical neural networks on those ten dots. One sees only the dots. The other also has to obey `m x'' + c x' + k x = 0` at 60 times where nothing was measured.

**Read the guide: https://jeyavardhan-l.github.io/Physics-informed-neural-network/**. It has interactive figures, including both networks training live in the page.

## Results

Mean squared error on 0.3 to 1 s, over ten seeds (each with its own noise and starting weights).

| Method | Median | Worst seed |
|---|---|---|
| Plain network | 1.6 | 3.4 |
| Always answer zero | 0.078 | 0.078 |
| **PINN** | **3.8e-6** | **3.3e-5** |
| Least squares with the exact formula | 2.6e-6 | 2.7e-5 |

- The PINN's error is about **21,000 times lower than always answering zero**, and about 440,000 times lower than the plain network's. The second ratio is larger and means less, because the plain network's error is nearly arbitrary.
- It is **1.4 times the error of least squares**, the best linear unbiased method for this data. On this problem least squares is also much faster. The guide says when a PINN is worth it and when it isn't.
- On a spring four times faster, the same setup reports a physics loss near `1e-7` and is wrong. The network obeys the equation at the points where it is checked and breaks it between them. [Chapter 8](https://jeyavardhan-l.github.io/Physics-informed-neural-network/breaks.html) measures this and fixes it.

## What is inside

| Path | What it is |
|---|---|
| `pinn/physics.py` | the oscillator and its exact solution |
| `pinn/derivatives.py`, `pinn/losses.py` | x' and x'' by autograd, the residual |
| `pinn/train.py`, `pinn/models.py` | Adam then L-BFGS; the plain net and the PINN |
| `pinn/inverse.py` | recover c and k from the dots |
| `pinn/classical.py` | least squares and the noise floor |
| `pinn/scratch.py` | the PINN in NumPy, every gradient derived by hand |
| `docs/assets/engine.js` | the same again in JavaScript, for the live figures |
| `experiments/`, `results/` | every experiment and its recorded numbers |
| `docs/`, `tools/guide/` | the field guide and the script that builds it |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux and macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q             # about two minutes
node --test tests/engine.test.mjs

python -m experiments.headline  # ten seeds, five methods
python -m experiments.ablations # one change at a time
python -m experiments.frequency
python -m experiments.inverse

python -m tools.guide.build     # rebuild docs/ from pages and results
python -m http.server -d docs 8000
```

## References

- Lagaris, Likas, Fotiadis. *Artificial neural networks for solving ordinary and partial differential equations.* IEEE Transactions on Neural Networks, 1998.
- Raissi, Perdikaris, Karniadakis. *Physics-informed neural networks.* Journal of Computational Physics, 2019.
- Krishnapriyan et al. *Characterizing possible failure modes in physics-informed neural networks.* NeurIPS, 2021.
- Wang, Teng, Perdikaris. *Understanding and mitigating gradient flow pathologies in physics-informed neural networks.* SIAM Journal on Scientific Computing, 2021.
- Moseley. *So, what is a physics-informed neural network?* 2021. The damped oscillator as a teaching example.
