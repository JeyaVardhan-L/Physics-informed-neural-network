window.SOURCE = [
{
"path": "pinn/physics.py",
"title": "The damped oscillator and its exact solution",
"group": "Core",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"The physical system: a mass on a spring with friction.",
"",
"    m x''(t) + c x'(t) + k x(t) = 0",
"",
"m is the mass (kg), c the damping (kg/s), k the stiffness (N/m). Everything",
"else in the project trusts this file, so it has the most direct tests.",
"\"\"\"",
"",
"from dataclasses import dataclass",
"",
"import numpy as np",
"",
""
],
"note": "<p>The whole project is about one equation. This file is the only place that knows its exact solution, and no network ever imports the solution for training.</p>"
},
{
"line": 14,
"code": [
"@dataclass(frozen=True)",
"class Oscillator:",
"    m: float = 1.0",
"    c: float = 3.0",
"    k: float = 225.0",
"    x0: float = 1.0  # position at t = 0",
"    v0: float = 0.0  # velocity at t = 0",
""
],
"note": "<p><code>frozen=True</code> makes an oscillator a value. Nothing can change its stiffness halfway through an experiment.</p><p><code>c = 3</code> and <code>k = 225</code> give a damping ratio of 0.1 and about two and a half swings in a second.</p>"
},
{
"line": 22,
"code": [
"    def __post_init__(self):",
"        if self.m <= 0 or self.k <= 0 or self.c < 0:",
"            raise ValueError(\"need m > 0, k > 0 and c >= 0\")",
"        if self.zeta >= 1:",
"            raise ValueError(",
"                f\"damping ratio {self.zeta:.2f} >= 1: the closed form here \"",
"                \"only covers the underdamped case\"",
"            )",
"",
"    @property"
],
"note": "<p>Bad parameters fail here, at construction. Without the <code>zeta</code> check, <code>omega_d</code> would be the square root of a negative number and everything after it would be <code>nan</code>.</p>"
},
{
"line": 32,
"code": [
"    def omega0(self):",
"        \"\"\"Natural frequency, the rate it would swing at without friction.\"\"\"",
"        return np.sqrt(self.k / self.m)",
"",
"    @property",
"    def delta(self):",
"        \"\"\"Decay rate of the envelope exp(-delta t).\"\"\"",
"        return self.c / (2 * self.m)",
"",
"    @property",
"    def zeta(self):",
"        \"\"\"Damping ratio. Below 1 the mass oscillates, at 1 or above it creeps.\"\"\"",
"        return self.c / (2 * np.sqrt(self.m * self.k))",
"",
"    @property",
"    def omega_d(self):",
"        \"\"\"The frequency it really swings at, slightly below omega0.\"\"\"",
"        return np.sqrt(self.omega0**2 - self.delta**2)",
""
],
"note": "<p>Four numbers that describe the motion. <i>omega0</i> is the frequency with no friction, <i>delta</i> the decay rate, <i>zeta</i> their ratio, <i>omega_d</i> the frequency you would actually measure.</p><p>They are properties, computed from m, c and k each time, so they can't get out of step.</p>"
},
{
"line": 51,
"code": [
"    def position(self, t):",
"        t = np.asarray(t, dtype=float)",
"        a, b = self._coefficients()",
"        w = self.omega_d",
"        return np.exp(-self.delta * t) * (a * np.cos(w * t) + b * np.sin(w * t))",
""
],
"note": "<p>The underdamped solution: a decaying exponential times a rotation.</p>"
},
{
"line": 57,
"code": [
"    def velocity(self, t):",
"        t = np.asarray(t, dtype=float)",
"        a, b = self._coefficients()",
"        w, d = self.omega_d, self.delta",
"        return np.exp(-d * t) * (",
"            (b * w - d * a) * np.cos(w * t) - (a * w + d * b) * np.sin(w * t)",
"        )",
""
],
"note": "<p>The product rule applied to <code>position</code>. Collect the cosine terms and the sine terms separately.</p>"
},
{
"line": 65,
"code": [
"    def acceleration(self, t):",
"        # Newton's second law. The equation itself says what x'' must be.",
"        return -(self.c * self.velocity(t) + self.k * self.position(t)) / self.m",
""
],
"note": "<p>Defined from the ODE and not from the formula. That makes the test meaningful: if <code>position</code> were not a solution, the numerical slope of <code>velocity</code> would disagree with this.</p>"
},
{
"line": 69,
"code": [
"    def energy(self, t):",
"        return 0.5 * self.m * self.velocity(t) ** 2 + 0.5 * self.k * self.position(t) ** 2",
""
],
"note": "<p>Kinetic plus spring energy. Friction can only remove it, so a test checks that it never rises.</p>"
},
{
"line": 72,
"code": [
"    def _coefficients(self):",
"        # a fixes x(0) = x0. b fixes x'(0) = v0, and is not zero even when",
"        # v0 = 0, because the envelope is already falling at t = 0.",
"        return self.x0, (self.v0 + self.delta * self.x0) / self.omega_d"
],
"note": "<p><code>b</code> has the extra <code>delta * x0</code>. Leave it out and <code>velocity(0)</code> is <i>minus delta</i> when it should be 0: the mass would start with a small downward kick.</p>"
}
]
},
{
"path": "pinn/data.py",
"title": "Ten noisy measurements and the scoring window",
"group": "Core",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"Ten noisy measurements, all taken early.",
"",
"The whole experiment rests on one split. Measurements exist only in",
"[0, T_DATA]. The question is asked on (T_DATA, T_END].",
"\"\"\"",
"",
"from dataclasses import dataclass",
"",
"import numpy as np",
"",
"T_DATA = 0.3",
"T_END = 1.0",
"N_POINTS = 10",
"NOISE = 0.01  # standard deviation, in the units of x. x(0) is 1, so this is 1%.",
"",
""
],
"note": "<p>The four constants are the experiment. <code>T_DATA</code> is the wall: data on the left, questions on the right.</p>"
},
{
"line": 17,
"code": [
"@dataclass(frozen=True)",
"class Measurements:",
"    t: np.ndarray",
"    x: np.ndarray",
"    noise: float",
"    seed: int",
"",
"    def __len__(self):",
"        return len(self.t)",
"",
""
],
"note": "<p>The seed and the noise level travel with the data, so a result can always say which dots it was trained on.</p>"
},
{
"line": 28,
"code": [
"def measure(osc, n=N_POINTS, t_max=T_DATA, noise=NOISE, seed=0):",
"    \"\"\"Sample the true motion at n evenly spaced times and add Gaussian noise.\"\"\"",
"    rng = np.random.default_rng(seed)",
"    t = np.linspace(0.0, t_max, n)",
"    x = osc.position(t) + noise * rng.standard_normal(n)",
"    return Measurements(t=t, x=x, noise=noise, seed=seed)",
"",
""
],
"note": "<p><code>default_rng(seed)</code> is a private random generator. Calling this can't disturb the random state of anything else, and the same seed always gives the same dots.</p>"
},
{
"line": 36,
"code": [
"def future_grid(n=500, start=T_DATA, stop=T_END):",
"    \"\"\"Times used for scoring. None of them is ever shown to a network.\"\"\"",
"    return np.linspace(start, stop, n + 1)[1:]"
],
"note": "<p><code>linspace(start, stop, n + 1)[1:]</code> drops the first point, so the grid starts just after the last dot. A dot at exactly 0.3 would otherwise be both training and test.</p>"
}
]
},
{
"path": "pinn/classical.py",
"title": "Least squares with the exact formula",
"group": "Core",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"What a physicist would do without any network.",
"",
"If m, c and k are known, every solution of the equation has the form",
"",
"    x(t) = a * exp(-delta t) cos(wd t) + b * exp(-delta t) sin(wd t)",
"",
"Only a and b are unknown and x is linear in both, so fitting ten dots is an",
"ordinary 2-parameter least-squares problem. This is the fair yardstick for the",
"PINN: it uses the same dots and the same equation, and it is optimal for",
"Gaussian noise.",
"\"\"\"",
"",
"import numpy as np",
"",
""
],
"note": "<p>A comparison with only a weak opponent proves little. This is the strong one: the same dots and the same physics, solved exactly.</p>"
},
{
"line": 16,
"code": [
"def basis(osc, t):",
"    t = np.asarray(t, dtype=float)",
"    env = np.exp(-osc.delta * t)",
"    return np.stack([env * np.cos(osc.omega_d * t), env * np.sin(osc.omega_d * t)], axis=1)",
"",
""
],
"note": "<p>One row per time, two columns: the decaying cosine and the decaying sine. Every solution of the equation is a mix of these two columns.</p>"
},
{
"line": 22,
"code": [
"def fit(osc, data):",
"    \"\"\"Return a function t -> x(t) with the two constants fitted to the dots.\"\"\"",
"    coef, *_ = np.linalg.lstsq(basis(osc, data.t), data.x, rcond=None)",
"    return lambda t: basis(osc, t) @ coef",
"",
""
],
"note": "<p>Two unknowns, ten equations, linear. <code>lstsq</code> uses the SVD and takes microseconds.</p>"
},
{
"line": 28,
"code": [
"def expected_future_mse(osc, data, t_future):",
"    \"\"\"Average error of that fit, over all the noise it could have seen.",
"",
"    Least squares turns noise of variance s^2 into coefficient covariance",
"    s^2 (B'B)^-1. Push that through the basis at the future times. No unbiased",
"    linear method using these dots can do better (Gauss-Markov).",
"    \"\"\"",
"    B, F = basis(osc, data.t), basis(osc, t_future)",
"    cov = data.noise**2 * np.linalg.inv(B.T @ B)",
"    return float(np.mean(np.sum((F @ cov) * F, axis=1)))"
],
"note": "<p>How much error should be expected from this fit, averaged over the noise? Noise of variance s² in the dots gives the coefficients a covariance of s²(BᵀB)⁻¹. Multiplying by the basis at future times turns that into a variance of the prediction.</p><p>The Gauss-Markov theorem says no linear unbiased estimator has a smaller variance. For Gaussian noise no unbiased estimator of any kind does.</p>"
}
]
},
{
"path": "pinn/network.py",
"title": "The multilayer perceptron",
"group": "Core",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"The network: one number in (time), one number out (position).",
"",
"Nothing in this file knows about springs. The plain net and the PINN are both",
"built here with the same seed, so they start with identical weights. The only",
"difference between them is the loss they are trained on.",
"\"\"\"",
"",
"import torch",
"from torch import nn",
"",
""
],
"note": "<p>The same class builds both the plain net and the PINN. Same seed, same weights. The comparison is then about the loss and nothing else.</p>"
},
{
"line": 12,
"code": [
"class Sine(nn.Module):",
"    def forward(self, z):",
"        return torch.sin(z)",
"",
"",
"ACTIVATIONS = {\"tanh\": nn.Tanh, \"sin\": Sine, \"softplus\": nn.Softplus, \"relu\": nn.ReLU}",
"",
""
],
"note": "<p>PyTorch has no sine activation module, so here is one.</p>"
},
{
"line": 20,
"code": [
"class MLP(nn.Module):",
"    def __init__(",
"        self,",
"        width=32,",
"        depth=3,",
"        activation=\"tanh\",",
"        input_scale=1.0,",
"        t_min=0.0,",
"        t_max=1.0,",
"        seed=0,",
"    ):",
"        super().__init__()",
"        self.t_min, self.t_max = t_min, t_max",
"        # Above 1, the first layer starts out with faster wiggles. See the",
"        # frequency experiment for when that is needed.",
"        self.input_scale = input_scale"
],
"note": "<p><code>input_scale</code> is 1 everywhere except in the frequency experiment.</p>"
},
{
"line": 36,
"code": [
"        generator = torch.Generator().manual_seed(seed)",
""
],
"note": "<p>A generator that belongs to this network. <code>torch.manual_seed</code> would work too, and would also reset the randomness of everything else in the program.</p>"
},
{
"line": 38,
"code": [
"        layers, size_in = [], 1",
"        for _ in range(depth):",
"            layers += [self._linear(size_in, width, generator), ACTIVATIONS[activation]()]",
"            size_in = width",
"        layers.append(self._linear(size_in, 1, generator))",
"        self.layers = nn.Sequential(*layers)",
""
],
"note": "<p>Linear, activation, linear, activation, ..., and a last linear layer with no activation, so the output can be any real number.</p>"
},
{
"line": 45,
"code": [
"    @staticmethod",
"    def _linear(size_in, size_out, generator):",
"        # float64 throughout. Second derivatives amplify rounding error, and",
"        # L-BFGS compares losses that differ in the seventh digit.",
"        # skip_init stops nn.Linear drawing its own default weights, which would",
"        # silently advance the global random state.",
"        layer = nn.utils.skip_init(nn.Linear, size_in, size_out, dtype=torch.float64)",
"        nn.init.xavier_uniform_(layer.weight, generator=generator)",
"        nn.init.zeros_(layer.bias)",
"        return layer",
""
],
"note": "<p><code>skip_init</code> creates the layer without drawing its default weights. Xavier initialisation then picks a range that keeps the size of signals about the same from layer to layer, which suits tanh.</p>"
},
{
"line": 56,
"code": [
"    def forward(self, t):",
"        # Map [t_min, t_max] to [-1, 1], where tanh is neither flat nor linear.",
"        # This happens inside forward so autograd differentiates with respect",
"        # to real seconds and the chain rule is applied for us.",
"        s = 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1",
"        return self.layers(self.input_scale * s)",
""
],
"note": "<p>Seconds in, position out. The rescaling is part of the function being differentiated, so <code>x'</code> comes out per second.</p>"
},
{
"line": 63,
"code": [
"    def n_parameters(self):",
"        return sum(p.numel() for p in self.parameters())",
"",
""
],
"note": "<p>Used by a test: 2,209 for the default size.</p>"
},
{
"line": 67,
"code": [
"def column(values):",
"    \"\"\"Turn a list or array of N numbers into the (N, 1) tensor a net expects.\"\"\"",
"    return torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1)"
],
"note": "<p>Networks take one row per sample. This turns a list of N times into an N by 1 tensor in float64.</p>"
}
]
},
{
"path": "pinn/derivatives.py",
"title": "Position, velocity and acceleration of a network",
"group": "Core",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"x, x' and x'' of a network with respect to its *input*.",
"",
"Ordinary training differentiates the loss with respect to the weights. A PINN",
"needs a second kind of derivative first: the output with respect to time.",
"\"\"\"",
"",
"import torch",
"",
""
],
"note": "<p>The heart of the method. Everything else in a PINN is ordinary.</p>"
},
{
"line": 10,
"code": [
"def derivatives(net, t):",
"    \"\"\"Return (x, v, a) at the times in t, all still attached to the graph.\"\"\"",
"    t = t.detach().clone().requires_grad_(True)",
"    x = net(t)",
"    v = _slope(x, t)",
"    a = _slope(v, t)",
"    return x, v, a",
"",
""
],
"note": "<p><code>detach().clone()</code> takes a private copy of the times, so turning on <code>requires_grad</code> can't leak back to the caller.</p><p><code>x</code>, <code>v</code> and <code>a</code> all stay attached to the graph. The loss will need to send gradients back through each of them.</p>"
},
{
"line": 19,
"code": [
"def _slope(y, t):",
"    # A straight line has a constant slope, and a constant has no graph to",
"    # differentiate. Its derivative is zero.",
"    if not y.requires_grad:",
"        return torch.zeros_like(t)"
],
"note": "<p>Added after a test failed. For a straight line, <code>v</code> is a constant with no graph, and asking autograd for its derivative raises <code>RuntimeError</code>.</p>"
},
{
"line": 24,
"code": [
"    # Each y[i] depends only on its own t[i], so summing y before differentiating",
"    # (which is what grad_outputs of ones does) still gives dy[i]/dt[i].",
"    #"
],
"note": "<p>The sum trick. The Jacobian of <code>y</code> with respect to <code>t</code> is diagonal, so multiplying it by a vector of ones gives the diagonal.</p>"
},
{
"line": 27,
"code": [
"    # create_graph=True keeps the derivative differentiable. Without it the loss",
"    # could not send gradients back through x'' to the weights.",
"    #",
"    # materialize_grads=True returns zeros where y does not depend on t at all.",
"    # That really happens: the second derivative of a ReLU net is zero.",
"    (dy,) = torch.autograd.grad(",
"        y, t, grad_outputs=torch.ones_like(y), create_graph=True, materialize_grads=True",
"    )",
"    return dy"
],
"note": "<p>The second derivative of a ReLU net doesn't depend on <code>t</code> at all. Without <code>materialize_grads=True</code>, autograd returns an error for an unused input. With it, it returns the honest answer: zeros.</p>"
}
]
},
{
"path": "pinn/losses.py",
"title": "Data loss and physics loss",
"group": "Core",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"The two terms of the loss.",
"",
"data_loss     how far the curve is from the dots",
"physics_loss  how badly the curve breaks m x'' + c x' + k x = 0",
"\"\"\"",
"",
"import torch",
"",
"from .derivatives import derivatives",
"",
""
],
"note": "<p>Two short functions and a scaling decision.</p>"
},
{
"line": 12,
"code": [
"def data_loss(net, t, x):",
"    return torch.mean((net(t) - x) ** 2)",
"",
""
],
"note": "<p>Ordinary mean squared error.</p>"
},
{
"line": 16,
"code": [
"def residual(net, t, m, c, k):",
"    \"\"\"What is left of the equation when the network is plugged in.",
"",
"    Dividing by k turns force into position. x is about 1 and k is 225, so",
"    without this the physics term would start around 225^2 = 50,000 times",
"    larger than the data term and the dots would be ignored.",
"    \"\"\"",
"    x, v, a = derivatives(net, t)",
"    return (m * a + c * v + k * x) / k",
"",
""
],
"note": "<p>Plug the network into the equation and return what is left over, one number per time.</p><p><code>m</code>, <code>c</code> and <code>k</code> are arguments. The inverse problem reuses the idea with <code>c</code> and <code>k</code> as trainable tensors.</p>"
},
{
"line": 27,
"code": [
"def physics_loss(net, t, m, c, k):",
"    return torch.mean(residual(net, t, m, c, k) ** 2)"
],
"note": "<p>Mean of the squared residual. It is zero when the curve solves the equation at every point given. \"At every point given\" is the fine print, and the frequency experiment is about that.</p>"
}
]
},
{
"path": "pinn/train.py",
"title": "Adam, then L-BFGS",
"group": "Core",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"Training: Adam to get into the right valley, L-BFGS to reach the bottom.\"\"\"",
"",
"import time",
"",
"import torch",
"",
""
],
"note": "<p>Two optimisers with different strengths.</p>"
},
{
"line": 8,
"code": [
"def fit(params, terms, weights, adam_steps=3000, lr=3e-3, lbfgs_steps=3000, log_every=100):",
"    \"\"\"Minimise a weighted sum of loss terms.",
"",
"    params   tensors to train",
"    terms    function returning {\"data\": tensor, \"physics\": tensor, ...}",
"    weights  {\"data\": 1.0, \"physics\": 0.1, ...}",
"    Returns a list of log rows, one per log_every steps.",
"    \"\"\"",
"    params = list(params)",
"    history, start = [], time.perf_counter()",
""
],
"note": "<p>It knows nothing about networks or springs. It takes tensors to train and a function that returns named loss terms. The plain net, the PINN and the inverse problem all use it.</p>"
},
{
"line": 19,
"code": [
"    def total(parts):",
"        return sum(weights[name] * value for name, value in parts.items())",
"",
"    def log(step, stage, parts):",
"        history.append(",
"            {",
"                \"step\": step,",
"                \"stage\": stage,",
"                \"seconds\": round(time.perf_counter() - start, 3),",
"                \"total\": float(total(parts).detach()),",
"                **{name: float(value.detach()) for name, value in parts.items()},",
"            }",
"        )",
""
],
"note": "<p>The weighted sum, kept apart from the terms so that the log records each term unweighted.</p>"
},
{
"line": 33,
"code": [
"    adam = torch.optim.Adam(params, lr=lr)",
"    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(adam, max(adam_steps, 1), eta_min=lr / 30)"
],
"note": "<p>Adam copes with a bad start: random weights, steep directions next to flat ones. The cosine schedule lowers the step size smoothly to 1/30 of its start, so it settles instead of bouncing.</p>"
},
{
"line": 35,
"code": [
"    for step in range(adam_steps):",
"        adam.zero_grad()",
"        parts = terms()",
"        total(parts).backward()",
"        if step % log_every == 0:",
"            log(step, \"adam\", parts)",
"        adam.step()",
"        schedule.step()",
""
],
"note": "<p>The gradient is cleared, the terms are computed, <code>backward()</code> fills the gradients and <code>step()</code> uses them. The log is written before the step, so row 0 is the untrained network.</p>"
},
{
"line": 44,
"code": [
"    if lbfgs_steps:",
"        lbfgs = torch.optim.LBFGS(",
"            params,",
"            max_iter=lbfgs_steps,",
"            max_eval=lbfgs_steps * 2,",
"            history_size=50,",
"            tolerance_grad=1e-12,",
"            tolerance_change=1e-16,",
"            line_search_fn=\"strong_wolfe\",",
"        )",
"        calls = 0",
""
],
"note": "<p>L-BFGS is a quasi-Newton method. It keeps the last 50 changes in position and gradient and uses them to estimate curvature. <code>strong_wolfe</code> is a line search that only accepts steps that lower the loss enough.</p><p>The tolerances are very tight. With the defaults it can stop early on losses this small.</p>"
},
{
"line": 56,
"code": [
"        def closure():",
"            # L-BFGS evaluates the loss several times per step while it searches",
"            # along a line, so it needs a function and not a single number.",
"            nonlocal calls",
"            lbfgs.zero_grad()",
"            parts = terms()",
"            loss = total(parts)",
"            loss.backward()",
"            if calls % log_every == 0:",
"                log(adam_steps + calls, \"lbfgs\", parts)",
"            calls += 1",
"            return loss",
""
],
"note": "<p>PyTorch's L-BFGS calls this many times inside a single <code>step()</code>, because the line search has to try several points.</p><p>It is full batch on purpose. If the loss changed randomly between calls, the curvature estimate would be noise.</p>"
},
{
"line": 69,
"code": [
"        lbfgs.step(closure)",
"        log(adam_steps + calls, \"lbfgs\", terms())",
"    else:",
"        log(adam_steps, \"adam\", terms())",
"    return history"
],
"note": "<p>One call runs up to <code>lbfgs_steps</code> iterations. The log row after it is the final state.</p>"
}
]
},
{
"path": "pinn/models.py",
"title": "The plain net and the PINN",
"group": "Core",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"The two contestants, built from the same parts.\"\"\"",
"",
"from dataclasses import dataclass, field",
"",
"import numpy as np",
"import torch",
"",
"from .data import T_DATA, T_END, future_grid",
"from .losses import data_loss, physics_loss",
"from .network import MLP, column",
"from .train import fit",
"",
""
],
"note": "<p>Read the two training functions next to each other. The difference is one entry in a dictionary.</p>"
},
{
"line": 14,
"code": [
"@dataclass",
"class Result:",
"    net: MLP",
"    history: list = field(repr=False)",
"",
"    def predict(self, t):",
"        with torch.no_grad():",
"            return self.net(column(t)).numpy().ravel()",
"",
""
],
"note": "<p>A trained network and its log. <code>predict</code> hides tensors and <code>no_grad</code> from everything outside.</p>"
},
{
"line": 24,
"code": [
"def train_plain(data, seed=0, width=32, depth=3, lbfgs_steps=0, **fit_options):",
"    \"\"\"The dots and nothing else.",
"",
"    L-BFGS is off by default because it hurts here. It drives the data loss",
"    far below the noise level, so it fits the noise, and the future gets worse.",
"    \"\"\"",
"    net = MLP(width=width, depth=depth, t_max=T_END, seed=seed)",
"    t, x = column(data.t), column(data.x)",
"    history = fit(",
"        net.parameters(),",
"        lambda: {\"data\": data_loss(net, t, x)},",
"        {\"data\": 1.0},",
"        lbfgs_steps=lbfgs_steps,",
"        **fit_options,",
"    )",
"    return Result(net, history)",
"",
""
],
"note": "<p><code>lbfgs_steps=0</code> by default. This gives the baseline its better score, because L-BFGS makes it worse.</p>"
},
{
"line": 42,
"code": [
"def train_pinn(",
"    data,",
"    osc,",
"    seed=0,",
"    width=32,",
"    depth=3,",
"    activation=\"tanh\",",
"    input_scale=1.0,",
"    n_collocation=60,",
"    t_physics=(0.0, T_END),",
"    weight=10.0,",
"    **fit_options,",
"):",
"    \"\"\"The dots, plus the equation at n_collocation times that have no dots.",
"",
"    osc is used for m, c and k only. Its x0 and v0, and so the true curve,",
"    never reach the loss.",
"    \"\"\""
],
"note": "<p>The defaults are the settings chosen on seeds 100 to 102: 60 collocation points, weight 10, Adam then L-BFGS.</p>"
},
{
"line": 60,
"code": [
"    net = MLP(",
"        width=width,",
"        depth=depth,",
"        activation=activation,",
"        input_scale=input_scale,",
"        t_max=T_END,",
"        seed=seed,",
"    )"
],
"note": "<p>The same class and the same seed as the plain net.</p>"
},
{
"line": 68,
"code": [
"    t, x = column(data.t), column(data.x)",
"    t_col = column(np.linspace(*t_physics, n_collocation))",
"    history = fit(",
"        net.parameters(),",
"        lambda: {",
"            \"data\": data_loss(net, t, x),",
"            \"physics\": physics_loss(net, t_col, osc.m, osc.c, osc.k),",
"        },",
"        {\"data\": 1.0, \"physics\": weight},",
"        **fit_options,",
"    )",
"    return Result(net, history)",
"",
""
],
"note": "<p>Collocation times cover <code>t_physics</code>, which is the full second unless an experiment says otherwise. They are fixed, so L-BFGS sees a loss that doesn't change between calls.</p>"
},
{
"line": 82,
"code": [
"def dense_residual(net, osc, n=2000):",
"    \"\"\"Physics loss at n times, most of which training never visited.",
"",
"    If this is far above the training value, the network has learned to obey",
"    the equation only at the points where it is checked.",
"    \"\"\"",
"    return float(physics_loss(net, column(np.linspace(0, T_END, n)), osc.m, osc.c, osc.k).detach())",
"",
""
],
"note": "<p>The audit. Training sees the residual at 60 times. This looks at 2,000. A large gap means the network is cheating between the points.</p>"
},
{
"line": 91,
"code": [
"def future_mse(predict, osc):",
"    \"\"\"Mean squared error on (T_DATA, T_END], against the noiseless truth.\"\"\"",
"    t = future_grid(start=T_DATA, stop=T_END)",
"    return float(np.mean((predict(t) - osc.position(t)) ** 2))"
],
"note": "<p>The score used for every method. It takes a function from times to positions, so it can score a network, a formula or <code>lambda t: 0 * t</code>.</p>"
}
]
},
{
"path": "pinn/inverse.py",
"title": "Finding c and k from the dots",
"group": "Core",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"Run the equation backwards: the dots are known, c and k are not.",
"",
"c and k become two more trainable numbers. They sit inside the residual, so the",
"same gradients that shape the curve also pull them towards values that make the",
"curve and the equation agree.",
"",
"m is fixed at 1. Multiplying m, c and k by the same number leaves the equation",
"unchanged, so motion alone can never reveal all three.",
"\"\"\"",
"",
"from dataclasses import dataclass, field",
"",
"import numpy as np",
"import torch",
"",
"from .data import T_END",
"from .derivatives import derivatives",
"from .losses import data_loss",
"from .models import Result",
"from .network import MLP, column",
"from .train import fit",
"",
""
],
"note": "<p>The forward problem knows the law and finds the curve. Here the form of the law is known and two of its numbers are not.</p><p>The scaling remark is an <i>identifiability</i> argument. It is worth having ready, because it is true of any amount of data.</p>"
},
{
"line": 24,
"code": [
"@dataclass",
"class Discovery(Result):",
"    c: float = 0.0",
"    k: float = 0.0",
"    trace: list = field(default_factory=list, repr=False)",
"",
""
],
"note": "<p>A <code>Result</code> with two extra numbers and the path they took.</p>"
},
{
"line": 31,
"code": [
"def discover(data, c_guess=1.0, k_guess=100.0, seed=0, n_collocation=60, weight=10.0, **fit_options):",
"    net = MLP(t_max=T_END, seed=seed)",
"    t, x = column(data.t), column(data.x)",
"    t_col = column(np.linspace(0, T_END, n_collocation))",
""
],
"note": "<p>The guesses are poor on purpose: c is off by a factor of 3 and k by more than 2.</p>"
},
{
"line": 36,
"code": [
"    # Train the logarithms. c and k then stay positive whatever the optimiser",
"    # does, and a step means \"a few percent\" for both even though k is 75 times",
"    # larger than c.",
"    log_c = torch.tensor(np.log(c_guess), dtype=torch.float64, requires_grad=True)",
"    log_k = torch.tensor(np.log(k_guess), dtype=torch.float64, requires_grad=True)",
"    trace, calls = [], 0",
""
],
"note": "<p>A change of variable that enforces positivity with no constraints. It also fixes the scale: Adam moves every parameter by about the same amount per step, and c and k differ by a factor of 75.</p>"
},
{
"line": 43,
"code": [
"    def terms():",
"        nonlocal calls",
"        c, k = log_c.exp(), log_k.exp()",
"        if calls % 50 == 0:",
"            trace.append([calls, float(c.detach()), float(k.detach())])",
"        calls += 1",
"        pos, vel, acc = derivatives(net, t_col)",
"        # Scaled by the fixed guess. Dividing by the trainable k would change the",
"        # loudness of the physics term while k is still being searched for."
],
"note": "<p>The same two loss terms as always. <code>c</code> and <code>k</code> are now tensors with gradients, so <code>backward()</code> reaches them too.</p>"
},
{
"line": 52,
"code": [
"        r = (acc + c * vel + k * pos) / k_guess",
"        return {\"data\": data_loss(net, t, x), \"physics\": torch.mean(r**2)}",
""
],
"note": "<p>Divided by a fixed number. Dividing by the trainable <code>k</code> would change the exchange rate between the two loss terms while <code>k</code> is being searched for.</p>"
},
{
"line": 55,
"code": [
"    history = fit(",
"        [*net.parameters(), log_c, log_k], terms, {\"data\": 1.0, \"physics\": weight}, **fit_options",
"    )",
"    c, k = float(log_c.detach().exp()), float(log_k.detach().exp())",
"    return Discovery(net, history, c=c, k=k, trace=trace)"
],
"note": "<p>The only change to the training call is two extra entries in the list of parameters.</p>"
}
]
},
{
"path": "pinn/scratch.py",
"title": "The PINN in NumPy, gradients by hand",
"group": "By hand",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"The same PINN with no autograd and no PyTorch, only NumPy.",
"",
"Two ideas, one for each direction.",
"",
"Forward   Carry three numbers through every layer: a value, its first time",
"          derivative and its second. (a, a1, a2) instead of only a.",
"",
"Backward  Walk the layers in reverse and hand each of those three numbers its",
"          share of the blame for the loss. The output of the walk is the",
"          gradient for every weight.",
"",
"The browser figure on the site runs a line-by-line port of this file, and the",
"tests compare its gradients with autograd's.",
"\"\"\"",
"",
"import numpy as np",
"",
""
],
"note": "<p>Chapter 10 derives every line of this file.</p>"
},
{
"line": 19,
"code": [
"def tanh_and_derivatives(z):",
"    \"\"\"tanh(z) and its first three derivatives.\"\"\"",
"    h = np.tanh(z)",
"    h1 = 1 - h**2",
"    h2 = -2 * h * h1",
"    h3 = -2 * h1 * (1 - 3 * h**2)",
"    return h, h1, h2, h3",
"",
""
],
"note": "<p>Each derivative is written in terms of the ones before it. The third is needed only on the way back.</p>"
},
{
"line": 28,
"code": [
"class ScratchMLP:",
"    def __init__(self, width=32, depth=3, t_max=1.0, seed=0):",
"        rng = np.random.default_rng(seed)",
"        sizes = [1] + [width] * depth + [1]",
"        self.t_max = t_max",
"        self.W, self.b = [], []",
"        for n_in, n_out in zip(sizes[:-1], sizes[1:]):",
"            bound = np.sqrt(6 / (n_in + n_out))  # Xavier uniform",
"            self.W.append(rng.uniform(-bound, bound, (n_out, n_in)))",
"            self.b.append(np.zeros(n_out))",
""
],
"note": "<p>Weights are stored the way PyTorch stores them, outputs by inputs, so a test can copy them across.</p>"
},
{
"line": 39,
"code": [
"    def forward(self, t):",
"        \"\"\"Return x, x', x'' at times t, and what backward() will need.\"\"\"",
"        t = np.asarray(t, dtype=float).reshape(-1, 1)",
"        # s = 2 t / t_max - 1, so ds/dt = 2 / t_max and d2s/dt2 = 0.",
"        a = 2 * t / self.t_max - 1",
"        a1 = np.full_like(t, 2 / self.t_max)",
"        a2 = np.zeros_like(t)",
""
],
"note": "<p>The three inputs: scaled time, its slope 2/T, and its second derivative, which is zero.</p>"
},
{
"line": 47,
"code": [
"        cache, last = [], len(self.W) - 1",
"        for i, (W, b) in enumerate(zip(self.W, self.b)):",
"            # A linear layer is the same matrix applied to all three. The bias",
"            # is a constant, so it disappears from both derivatives.",
"            z, z1, z2 = a @ W.T + b, a1 @ W.T, a2 @ W.T",
"            cache.append((a, a1, a2, z, z1, z2))",
"            if i == last:",
"                a, a1, a2 = z, z1, z2",
"            else:",
"                h, h1, h2, _ = tanh_and_derivatives(z)",
"                a = h",
"                a1 = h1 * z1  # chain rule",
"                a2 = h2 * z1**2 + h1 * z2  # chain rule again, then product rule",
"        return a, a1, a2, cache",
""
],
"note": "<p>The bias appears once. <code>W</code> appears three times, which is why its gradient will have three terms.</p><p>The last layer is linear, so it skips the activation.</p>"
},
{
"line": 62,
"code": [
"    def backward(self, cache, g, g1, g2):",
"        \"\"\"Gradients of a loss, given its derivative by x, x' and x''.",
"",
"        g is dL/dx, g1 is dL/dx', g2 is dL/dx''. Each has shape (N, 1).",
"        \"\"\"",
"        dW, db = [None] * len(self.W), [None] * len(self.b)",
"        last = len(self.W) - 1",
"        for i in range(last, -1, -1):",
"            a, a1, a2, z, z1, z2 = cache[i]",
"            if i != last:"
],
"note": "<p><code>g</code>, <code>g1</code> and <code>g2</code> are the loss's derivatives with respect to this layer's three outputs. They have the same shapes as the outputs.</p>"
},
{
"line": 72,
"code": [
"                _, h1, h2, h3 = tanh_and_derivatives(z)",
"                # z appears in all three outputs, z1 in two of them, z2 in one.",
"                gz = g * h1 + g1 * h2 * z1 + g2 * (h3 * z1**2 + h2 * z2)",
"                gz1 = g1 * h1 + g2 * 2 * h2 * z1",
"                gz2 = g2 * h1",
"            else:",
"                gz, gz1, gz2 = g, g1, g2"
],
"note": "<p>Ask of each input: where was it used? <code>z2</code> in one place, <code>z1</code> in two, <code>z</code> in three. Each use contributes one term.</p>"
},
{
"line": 79,
"code": [
"            dW[i] = gz.T @ a + gz1.T @ a1 + gz2.T @ a2",
"            db[i] = gz.sum(axis=0)",
"            W = self.W[i]",
"            g, g1, g2 = gz @ W, gz1 @ W, gz2 @ W",
"        return dW, db",
""
],
"note": "<p>Three uses of <code>W</code>, three outer products. <code>@</code> sums over the batch at the same time.</p>"
},
{
"line": 85,
"code": [
"    def loss_and_grads(self, t_data, x_data, t_col, m, c, k, weight=10.0):",
"        x_data = np.asarray(x_data, dtype=float).reshape(-1, 1)",
"",
"        # Data term: mean((x - dots)^2). Only x matters, so g1 = g2 = 0.",
"        x, _, _, cache = self.forward(t_data)",
"        err = x - x_data",
"        zero = np.zeros_like(err)",
"        dW_d, db_d = self.backward(cache, 2 * err / len(err), zero, zero)",
""
],
"note": "<p>Two passes, one for the dots and one for the collocation points. Gradients add, so the results are added at the end.</p>"
},
{
"line": 94,
"code": [
"        # Physics term: mean(r^2) with r = (m x'' + c x' + k x) / k.",
"        x, v, a, cache = self.forward(t_col)",
"        r = (m * a + c * v + k * x) / k",
"        g = weight * 2 * r / len(r)",
"        dW_p, db_p = self.backward(cache, g, g * c / k, g * m / k)",
"",
"        parts = {\"data\": float(np.mean(err**2)), \"physics\": float(np.mean(r**2))}",
"        grads = [p + q for p, q in zip(dW_d + db_d, dW_p + db_p)]",
"        return parts, grads",
"",
"    def parameters(self):",
"        return self.W + self.b",
"",
""
],
"note": "<p>The chain rule through the residual: dr/dx = 1, dr/dx' = c/k and dr/dx'' = m/k. These three numbers are the only place where physics enters the gradient.</p>"
},
{
"line": 108,
"code": [
"class Adam:",
"    def __init__(self, params, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):",
"        self.params, self.lr, self.beta1, self.beta2, self.eps = params, lr, beta1, beta2, eps",
"        self.mean = [np.zeros_like(p) for p in params]",
"        self.var = [np.zeros_like(p) for p in params]",
"        self.t = 0",
"",
"    def step(self, grads):",
"        self.t += 1",
"        for p, g, mean, var in zip(self.params, grads, self.mean, self.var):",
"            mean += (1 - self.beta1) * (g - mean)",
"            var += (1 - self.beta2) * (g**2 - var)",
"            mean_hat = mean / (1 - self.beta1**self.t)",
"            var_hat = var / (1 - self.beta2**self.t)",
"            p -= self.lr * mean_hat / (np.sqrt(var_hat) + self.eps)  # in place",
"",
""
],
"note": "<p>A running mean of the gradient and of its square. The step is their ratio, so its size depends little on the scale of the gradient. The hats correct for both averages starting at zero.</p>"
},
{
"line": 125,
"code": [
"def train(net, data, osc, steps=15000, lr=3e-3, weight=10.0, n_collocation=60, t_end=1.0):",
"    adam, t_col, history = Adam(net.parameters(), lr), np.linspace(0, t_end, n_collocation), []",
"    for step in range(steps):",
"        adam.lr = lr / 30 + (lr - lr / 30) * 0.5 * (1 + np.cos(np.pi * step / steps))",
"        parts, grads = net.loss_and_grads(data.t, data.x, t_col, osc.m, osc.c, osc.k, weight)",
"        adam.step(grads)",
"        if step % 500 == 0:",
"            history.append({\"step\": step, **parts})",
"    return history"
],
"note": "<p>The same cosine schedule as the PyTorch version. No L-BFGS here.</p>"
}
]
},
{
"path": "docs/assets/engine.js",
"title": "The PINN that runs in the page",
"group": "By hand",
"blocks": [
{
"line": 1,
"code": [
"/* A PINN that trains in the browser. Port of pinn/scratch.py, no libraries.",
"   tests/engine.test.mjs compares its gradients with the NumPy ones. */",
"(function (root) {",
"  \"use strict\";",
""
],
"note": "<p>A port of <code>pinn/scratch.py</code>. No libraries, so it can be read from top to bottom.</p>"
},
{
"line": 6,
"code": [
"  /* ---- physics: same closed form as pinn/physics.py ---- */",
"  function oscillator(m, c, k, x0, v0) {",
"    x0 = x0 === undefined ? 1 : x0; v0 = v0 || 0;",
"    var d = c / (2 * m), w = Math.sqrt(k / m - d * d), b = (v0 + d * x0) / w;",
"    return {",
"      m: m, c: c, k: k, delta: d, omegaD: w, omega0: Math.sqrt(k / m),",
"      zeta: c / (2 * Math.sqrt(m * k)),",
"      x: function (t) { return Math.exp(-d * t) * (x0 * Math.cos(w * t) + b * Math.sin(w * t)); },",
"      v: function (t) {",
"        return Math.exp(-d * t) * ((b * w - d * x0) * Math.cos(w * t) - (x0 * w + d * b) * Math.sin(w * t));",
"      },",
"      a: function (t) { return -(c * this.v(t) + k * this.x(t)) / m; }",
"    };",
"  }",
""
],
"note": "<p>The same formula as <code>pinn/physics.py</code>.</p>"
},
{
"line": 21,
"code": [
"  /* ---- small seeded random generator (mulberry32) ---- */",
"  function rng(seed) {",
"    var s = seed >>> 0;",
"    function u() {",
"      s = (s + 0x6D2B79F5) >>> 0;",
"      var t = Math.imul(s ^ (s >>> 15), 1 | s);",
"      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;",
"      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;",
"    }",
"    return {",
"      uniform: function (a, b) { return a + (b - a) * u(); },",
"      normal: function () {",
"        return Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());",
"      }",
"    };",
"  }",
""
],
"note": "<p>JavaScript's <code>Math.random</code> can't be seeded. This small generator can, so \"seed 1\" means the same dots on every visit.</p>"
},
{
"line": 38,
"code": [
"  function linspace(a, b, n) {",
"    var out = [];",
"    for (var i = 0; i < n; i++) out.push(n === 1 ? a : a + (b - a) * i / (n - 1));",
"    return out;",
"  }",
"",
"  function measure(osc, opts) {",
"    opts = opts || {};",
"    var r = rng(opts.seed || 0), noise = opts.noise === undefined ? 0.01 : opts.noise;",
"    var t = linspace(0, opts.tMax || 0.3, opts.n || 10);",
"    return { t: t, x: t.map(function (q) { return osc.x(q) + noise * r.normal(); }) };",
"  }",
""
],
"note": "<p>Named after NumPy's, with the same end points.</p>"
},
{
"line": 51,
"code": [
"  /* ---- the network ---- */",
"  function Net(opts) {",
"    opts = opts || {};",
"    var width = opts.width || 24, depth = opts.depth || 3, r = rng(opts.seed || 1);",
"    this.tMax = opts.tMax || 1;",
"    this.sizes = [1];",
"    for (var d = 0; d < depth; d++) this.sizes.push(width);",
"    this.sizes.push(1);",
"    this.W = []; this.b = []; this.gW = []; this.gb = []; this.buf = [];",
"    for (var l = 0; l + 1 < this.sizes.length; l++) {",
"      var nIn = this.sizes[l], nOut = this.sizes[l + 1], bound = Math.sqrt(6 / (nIn + nOut));",
"      var W = new Float64Array(nIn * nOut);",
"      for (var i = 0; i < W.length; i++) W[i] = r.uniform(-bound, bound);",
"      this.W.push(W); this.b.push(new Float64Array(nOut));",
"      this.gW.push(new Float64Array(nIn * nOut)); this.gb.push(new Float64Array(nOut));",
"      this.buf.push({",
"        a: new Float64Array(nIn), a1: new Float64Array(nIn), a2: new Float64Array(nIn),",
"        z: new Float64Array(nOut), z1: new Float64Array(nOut), z2: new Float64Array(nOut),",
"        g: new Float64Array(nOut), g1: new Float64Array(nOut), g2: new Float64Array(nOut)",
"      });",
"    }",
"  }",
""
],
"note": "<p>Weights live in flat <code>Float64Array</code>s. Buffers are allocated once and reused, because allocating inside the training loop would keep the garbage collector busy.</p>"
},
{
"line": 74,
"code": [
"  /* One time in; x, x' and x'' out. Intermediate values stay in this.buf. */",
"  Net.prototype.forward = function (t) {",
"    var L = this.W.length, B = this.buf, i, j, l;",
"    B[0].a[0] = 2 * t / this.tMax - 1; B[0].a1[0] = 2 / this.tMax; B[0].a2[0] = 0;",
"    for (l = 0; l < L; l++) {",
"      var W = this.W[l], bias = this.b[l], p = B[l], nIn = this.sizes[l], nOut = this.sizes[l + 1];",
"      var next = l + 1 < L ? B[l + 1] : null;",
"      for (j = 0; j < nOut; j++) {",
"        var z = bias[j], z1 = 0, z2 = 0, o = j * nIn;",
"        for (i = 0; i < nIn; i++) {",
"          var w = W[o + i];",
"          z += w * p.a[i]; z1 += w * p.a1[i]; z2 += w * p.a2[i];",
"        }",
"        p.z[j] = z; p.z1[j] = z1; p.z2[j] = z2;",
"        if (next) {",
"          var h = Math.tanh(z), h1 = 1 - h * h, h2 = -2 * h * h1;",
"          next.a[j] = h;",
"          next.a1[j] = h1 * z1;",
"          next.a2[j] = h2 * z1 * z1 + h1 * z2;",
"        }",
"      }",
"    }",
"    var q = B[L - 1];",
"    return [q.z[0], q.z1[0], q.z2[0]];",
"  };",
""
],
"note": "<p>One sample at a time. NumPy works on the whole batch with matrix products, and here the loops are explicit. The arithmetic is identical.</p>"
},
{
"line": 100,
"code": [
"  /* Add this point's gradients. g = dL/dx, g1 = dL/dx', g2 = dL/dx''. */",
"  Net.prototype.backward = function (g, g1, g2) {",
"    var L = this.W.length, B = this.buf, i, j, l;",
"    B[L - 1].g[0] = g; B[L - 1].g1[0] = g1; B[L - 1].g2[0] = g2;",
"    for (l = L - 1; l >= 0; l--) {",
"      var W = this.W[l], gW = this.gW[l], gb = this.gb[l], p = B[l];",
"      var nIn = this.sizes[l], nOut = this.sizes[l + 1], prev = l > 0 ? B[l - 1] : null;",
"      if (prev) { prev.g.fill(0); prev.g1.fill(0); prev.g2.fill(0); }",
"      for (j = 0; j < nOut; j++) {",
"        var gz = p.g[j], gz1 = p.g1[j], gz2 = p.g2[j], o = j * nIn;",
"        if (l < L - 1) {",
"          var z1 = p.z1[j], z2 = p.z2[j], h = Math.tanh(p.z[j]);",
"          var h1 = 1 - h * h, h2 = -2 * h * h1, h3 = -2 * h1 * (1 - 3 * h * h);",
"          var u = gz * h1 + gz1 * h2 * z1 + gz2 * (h3 * z1 * z1 + h2 * z2);",
"          var u1 = gz1 * h1 + gz2 * 2 * h2 * z1;",
"          gz2 = gz2 * h1; gz1 = u1; gz = u;",
"        }",
"        gb[j] += gz;",
"        for (i = 0; i < nIn; i++) {",
"          gW[o + i] += gz * p.a[i] + gz1 * p.a1[i] + gz2 * p.a2[i];",
"          if (prev) {",
"            var w = W[o + i];",
"            prev.g[i] += gz * w; prev.g1[i] += gz1 * w; prev.g2[i] += gz2 * w;",
"          }",
"        }",
"      }",
"    }",
"  };",
""
],
"note": "<p>The same three adjoints as in the NumPy version. <code>+=</code> accumulates over samples.</p>"
},
{
"line": 129,
"code": [
"  Net.prototype.zeroGrad = function () {",
"    this.gW.forEach(function (a) { a.fill(0); });",
"    this.gb.forEach(function (a) { a.fill(0); });",
"  };",
"",
"  Net.prototype.predict = function (t) { return this.forward(t)[0]; };",
"",
"  Net.prototype.nParameters = function () {",
"    var n = 0;",
"    this.W.concat(this.b).forEach(function (a) { n += a.length; });",
"    return n;",
"  };",
""
],
"note": "<p>Gradients accumulate, so they must be cleared before each evaluation.</p>"
},
{
"line": 142,
"code": [
"  /* mean((x - dots)^2) + weight * mean(r^2), and its gradient. */",
"  Net.prototype.lossAndGrads = function (data, tCol, phys, weight) {",
"    this.zeroGrad();",
"    var dataLoss = 0, physLoss = 0, n = data.t.length, i, out;",
"    for (i = 0; i < n; i++) {",
"      out = this.forward(data.t[i]);",
"      var e = out[0] - data.x[i];",
"      dataLoss += e * e / n;",
"      this.backward(2 * e / n, 0, 0);",
"    }",
"    if (weight > 0) {",
"      var N = tCol.length;",
"      for (i = 0; i < N; i++) {",
"        out = this.forward(tCol[i]);",
"        var r = (phys.m * out[2] + phys.c * out[1] + phys.k * out[0]) / phys.k;",
"        physLoss += r * r / N;",
"        var g = weight * 2 * r / N;",
"        this.backward(g, g * phys.c / phys.k, g * phys.m / phys.k);",
"      }",
"    }",
"    return { data: dataLoss, physics: physLoss, total: dataLoss + weight * physLoss };",
"  };",
""
],
"note": "<p>With <code>weight = 0</code> the second half is skipped and this is the plain net.</p>"
},
{
"line": 165,
"code": [
"  /* ---- Adam ---- */",
"  function Adam(net, lr) {",
"    this.lr = lr || 3e-3; this.t = 0;",
"    this.params = net.W.concat(net.b); this.grads = net.gW.concat(net.gb);",
"    this.mean = this.params.map(function (p) { return new Float64Array(p.length); });",
"    this.vari = this.params.map(function (p) { return new Float64Array(p.length); });",
"  }",
"  Adam.prototype.step = function () {",
"    this.t++;",
"    var c1 = 1 - Math.pow(0.9, this.t), c2 = 1 - Math.pow(0.999, this.t);",
"    for (var k = 0; k < this.params.length; k++) {",
"      var p = this.params[k], g = this.grads[k], m = this.mean[k], v = this.vari[k];",
"      for (var i = 0; i < p.length; i++) {",
"        m[i] += 0.1 * (g[i] - m[i]);",
"        v[i] += 0.001 * (g[i] * g[i] - v[i]);",
"        p[i] -= this.lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);",
"      }",
"    }",
"  };",
""
],
"note": "<p>The same update as <code>scratch.py</code>.</p>"
},
{
"line": 185,
"code": [
"  /* A training run that advances a few steps at a time, so a page can draw. */",
"  function Trainer(opts) {",
"    this.osc = opts.osc; this.data = opts.data;",
"    this.weight = opts.weight === undefined ? 10 : opts.weight;",
"    this.phys = opts.phys || opts.osc;",
"    this.tCol = opts.tCol || linspace(0, 1, opts.nCol || 60);",
"    this.steps = opts.steps || 4000; this.lr = opts.lr || 3e-3;",
"    this.net = new Net({ width: opts.width, depth: opts.depth, seed: opts.seed });",
"    this.adam = new Adam(this.net, this.lr);",
"    this.step = 0; this.history = []; this.parts = null;",
"  }",
"  Trainer.prototype.advance = function (n) {",
"    while (n-- > 0 && this.step < this.steps) {",
"      var lo = this.lr / 30;",
"      this.adam.lr = lo + (this.lr - lo) * 0.5 * (1 + Math.cos(Math.PI * this.step / this.steps));",
"      this.parts = this.net.lossAndGrads(this.data, this.tCol, this.phys, this.weight);",
"      if (this.step % 20 === 0) this.history.push({ step: this.step, data: this.parts.data, physics: this.parts.physics });",
"      this.adam.step();",
"      this.step++;",
"    }",
"    return this.step < this.steps;",
"  };",
"  Trainer.prototype.futureMse = function (from) {",
"    var s = 0, n = 200, a = from === undefined ? 0.3 : from;",
"    for (var i = 1; i <= n; i++) {",
"      var t = a + (1 - a) * i / n, e = this.net.predict(t) - this.osc.x(t);",
"      s += e * e / n;",
"    }",
"    return s;",
"  };",
""
],
"note": "<p><code>advance(n)</code> does a few steps and returns, so the page can draw a frame and stay responsive.</p>"
},
{
"line": 216,
"code": [
"  var api = { oscillator: oscillator, rng: rng, linspace: linspace, measure: measure, Net: Net, Adam: Adam, Trainer: Trainer };",
"  if (typeof module !== \"undefined\" && module.exports) module.exports = api;",
"  else root.Engine = api;",
"})(typeof self !== \"undefined\" ? self : this);"
],
"note": "<p>Works as a browser script and as a Node module, which is how the test loads it.</p>"
}
]
},
{
"path": "experiments/common.py",
"title": "Parallel runs and saved results",
"group": "Experiments",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"Shared plumbing: run jobs in parallel, save rows as JSON.\"\"\"",
"",
"import json",
"import os",
"from concurrent.futures import ProcessPoolExecutor",
"from pathlib import Path",
"",
"import numpy as np",
"",
"from pinn.data import T_END",
""
],
"note": "<p>Every number in the guide comes out of a JSON file written by <code>save</code>.</p>"
},
{
"line": 12,
"code": [
"RESULTS = Path(__file__).resolve().parent.parent / \"results\"",
"",
"# Seeds 100, 101 and 102 were used to choose weight = 10 and the optimiser",
"# settings. The seeds below were first run after those choices were fixed.",
"SEEDS = list(range(10))",
"FEW_SEEDS = [0, 1, 2]",
"",
"CURVE_T = np.linspace(0, T_END, 201)",
"",
""
],
"note": "<p>The seeds used for tuning and the seeds used for reporting don't overlap.</p>"
},
{
"line": 22,
"code": [
"def curve(predict):",
"    return [round(float(x), 5) for x in predict(CURVE_T)]",
"",
""
],
"note": "<p>Predictions on a fixed grid, rounded, for the figures.</p>"
},
{
"line": 26,
"code": [
"def run(job, arguments, workers=None):",
"    \"\"\"Call job(*args) for every args tuple, one process each, in order.\"\"\"",
"    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 2))",
"    with ProcessPoolExecutor(workers, initializer=_single_thread) as pool:",
"        futures = [pool.submit(job, *args) for args in arguments]",
"        rows = []",
"        for future in futures:",
"            rows.append(future.result())",
"            shown = {k: v for k, v in rows[-1].items() if not isinstance(v, (list, dict))}",
"            print(shown, flush=True)",
"    return rows",
"",
""
],
"note": "<p>One process per run. Futures are collected in order, so the rows come out in the same order every time.</p>"
},
{
"line": 39,
"code": [
"def _single_thread():",
"    # The tensors are tiny. Several threads per process only fight each other.",
"    import torch",
"",
"    torch.set_num_threads(1)",
"",
""
],
"note": "<p>Eight processes with twelve threads each would spend their time fighting over the same cores. The tensors are far too small to benefit.</p>"
},
{
"line": 46,
"code": [
"def save(name, payload):",
"    RESULTS.mkdir(exist_ok=True)",
"    path = RESULTS / f\"{name}.json\"",
"    path.write_text(json.dumps(payload, indent=1) + \"\\n\", encoding=\"utf-8\", newline=\"\\n\")",
"    print(\"wrote\", path.name)",
"",
""
],
"note": "<p>Newlines are fixed to <code>\\n</code>, so the files are the same on Windows and Linux.</p>"
},
{
"line": 53,
"code": [
"def median(rows, key=\"mse\", **match):",
"    values = [r[key] for r in rows if all(r[k] == v for k, v in match.items())]",
"    return float(np.median(values))"
],
"note": "<p>Median, because the errors span several orders of magnitude and a mean would report only the worst seed.</p>"
}
]
},
{
"path": "experiments/headline.py",
"title": "Five methods, ten seeds",
"group": "Experiments",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"Plain net against PINN, ten seeds, against two yardsticks.",
"",
"    python -m experiments.headline",
"\"\"\"",
"",
"from pinn import classical",
"from pinn.data import future_grid, measure",
"from pinn.models import future_mse, train_pinn, train_plain",
"from pinn.physics import Oscillator",
"",
"from .common import CURVE_T, SEEDS, curve, median, run, save",
"",
"OSC = Oscillator()",
"",
""
],
"note": "<p>The main result.</p>"
},
{
"line": 16,
"code": [
"def job(method, seed):",
"    data = measure(OSC, seed=seed)",
"    history = []",
"    if method == \"plain\":",
"        result = train_plain(data, seed=seed)",
"        predict, history = result.predict, result.history",
"    elif method == \"plain_lbfgs\":",
"        result = train_plain(data, seed=seed, lbfgs_steps=3000)",
"        predict, history = result.predict, result.history",
"    elif method == \"pinn\":",
"        result = train_pinn(data, OSC, seed=seed)",
"        predict, history = result.predict, result.history",
"    elif method == \"classical\":",
"        predict = classical.fit(OSC, data)",
"    elif method == \"zero\":",
"        predict = lambda t: 0 * t  # noqa: E731  (a net that says \"it has stopped\")"
],
"note": "<p>One method, one seed. Each seed gets its own noise and its own starting weights, so the spread covers both.</p>"
},
{
"line": 32,
"code": [
"    row = {\"method\": method, \"seed\": seed, \"mse\": future_mse(predict, OSC)}",
"    if history:",
"        row[\"data_loss\"] = history[-1][\"data\"]",
"        row[\"seconds\"] = history[-1][\"seconds\"]",
"    if seed < 3:",
"        row[\"curve\"] = curve(predict)",
"        row[\"history\"] = history",
"        row[\"dots\"] = {\"t\": data.t.tolist(), \"x\": data.x.tolist()}",
"    return row",
"",
""
],
"note": "<p>Full curves are saved only for seeds 0 to 2, which keeps the file small.</p>"
},
{
"line": 43,
"code": [
"if __name__ == \"__main__\":",
"    methods = [\"plain\", \"plain_lbfgs\", \"pinn\", \"classical\", \"zero\"]",
"    rows = run(job, [(m, s) for m in methods for s in SEEDS])",
"    floor = classical.expected_future_mse(OSC, measure(OSC), future_grid())",
"    summary = {m: median(rows, method=m) for m in methods}",
"    by_seed = {m: {r[\"seed\"]: r[\"mse\"] for r in rows if r[\"method\"] == m} for m in methods}",
"    ratios = sorted(by_seed[\"plain\"][s] / by_seed[\"pinn\"][s] for s in SEEDS)",
"    save(",
"        \"headline\",",
"        {",
"            \"median_mse\": summary,",
"            \"noise_floor\": floor,",
"            \"ratio_of_medians\": summary[\"plain\"] / summary[\"pinn\"],",
"            \"ratio_per_seed\": ratios,",
"            \"t\": CURVE_T.round(4).tolist(),",
"            \"truth\": curve(OSC.position),",
"            \"rows\": rows,",
"        },",
"    )",
"    print(summary, \"floor\", floor, \"ratio\", summary[\"plain\"] / summary[\"pinn\"])"
],
"note": "<p>Two ratios are saved: the ratio of medians and the ratio for each seed. They answer different questions and they don't have to agree.</p>"
}
]
},
{
"path": "experiments/ablations.py",
"title": "Change one thing at a time",
"group": "Experiments",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"Change one thing at a time and measure what it was doing.",
"",
"    python -m experiments.ablations             all of them",
"    python -m experiments.ablations coverage    just one",
"\"\"\"",
"",
"import sys",
"",
"from pinn import classical",
"from pinn.data import measure",
"from pinn.models import dense_residual, future_mse, train_pinn",
"from pinn.physics import Oscillator",
"",
"from .common import FEW_SEEDS, curve, run, save",
"",
"OSC = Oscillator()",
"",
""
],
"note": "<p>An <i>ablation</i> removes or changes one part and measures what was lost.</p>"
},
{
"line": 19,
"code": [
"def job(setting, seed, options, noise, believed, osc=OSC):",
"    \"\"\"believed holds the m, c, k the loss is told. Usually the true ones.\"\"\"",
"    data = measure(osc, seed=seed, noise=noise)",
"    result = train_pinn(data, believed, seed=seed, **options)",
"    last = result.history[-1]",
"    row = {",
"        \"setting\": setting,",
"        \"seed\": seed,",
"        \"mse\": future_mse(result.predict, osc),",
"        \"classical\": future_mse(classical.fit(osc, data), osc),",
"        \"data_loss\": last[\"data\"],",
"        \"physics_loss\": last[\"physics\"],",
"        \"dense_residual\": dense_residual(result.net, believed),",
"        \"seconds\": last[\"seconds\"],",
"    }",
"    if seed == 0:",
"        row[\"curve\"] = curve(result.predict)",
"    return row",
"",
""
],
"note": "<p><code>believed</code> is the physics the loss is told, and <code>osc</code> is the physics that made the dots. Keeping them apart is what makes the wrong-physics experiment possible.</p>"
},
{
"line": 39,
"code": [
"def variants():",
"    adam = lambda n: dict(adam_steps=n, lbfgs_steps=0)  # noqa: E731",
"    return {",
"        # How loud is the equation compared with the dots?",
"        \"weight\": [(w, dict(weight=w), 0.01, OSC) for w in (0.001, 0.1, 1.0, 10.0, 1000.0)],",
"        # Where is the equation enforced? Always 60 points, spread up to t.",
"        \"coverage\": [(t, dict(t_physics=(0.0, t)), 0.01, OSC) for t in (0.3, 0.5, 0.75, 1.0)],",
"        # At how many times is it enforced?",
"        \"collocation\": [(n, dict(n_collocation=n), 0.01, OSC) for n in (4, 6, 8, 12, 20, 60)],",
"        # Which optimiser does the work?",
"        \"optimiser\": [",
"            (\"adam 3000\", adam(3000), 0.01, OSC),",
"            (\"adam 6000\", adam(6000), 0.01, OSC),",
"            (\"adam 3000 + lbfgs\", {}, 0.01, OSC),",
"            (\"lbfgs only\", dict(adam_steps=0), 0.01, OSC),",
"        ],",
"        # Does the activation need a second derivative?",
"        \"activation\": [(a, dict(activation=a), 0.01, OSC) for a in (\"tanh\", \"sin\", \"softplus\", \"relu\")],",
"        # How far does it degrade as the dots get worse?",
"        \"noise\": [(s, {}, s, OSC) for s in (0.0, 0.003, 0.01, 0.03, 0.1)],",
"        # What if the equation is wrong?",
"        \"wrong_physics\": [",
"            (\"right\", {}, 0.01, OSC),",
"            (\"k 5% high\", {}, 0.01, Oscillator(k=1.05 * OSC.k)),",
"            (\"k 20% high\", {}, 0.01, Oscillator(k=1.2 * OSC.k)),",
"            (\"no friction\", {}, 0.01, Oscillator(c=0.0)),",
"        ],",
"    }",
"",
""
],
"note": "<p>A table of settings. Each entry is a label, the options for <code>train_pinn</code>, the noise and the believed physics.</p>"
},
{
"line": 69,
"code": [
"if __name__ == \"__main__\":",
"    chosen = sys.argv[1:] or list(variants())",
"    for name in chosen:",
"        print(\"==\", name)",
"        jobs = [",
"            (setting, seed, options, noise, believed)",
"            for setting, options, noise, believed in variants()[name]",
"            for seed in FEW_SEEDS",
"        ]",
"        save(f\"ablation_{name}\", run(job, jobs))"
],
"note": "<p>Three seeds per setting. That is enough to see large effects and too few to rank close ones.</p>"
}
]
},
{
"path": "experiments/frequency.py",
"title": "When the training loss lies",
"group": "Experiments",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"A faster spring, where the training loss stops being trustworthy.",
"",
"    python -m experiments.frequency",
"\"\"\"",
"",
"from pinn.physics import Oscillator",
"",
"from .ablations import job",
"from .common import FEW_SEEDS, curve, run, save",
"",
"SPRINGS = {15: Oscillator(k=225.0), 30: Oscillator(k=900.0), 60: Oscillator(k=3600.0)}",
""
],
"note": "<p>Only the stiffness changes: 225, 900, 3600. The frequency doubles each time.</p>"
},
{
"line": 13,
"code": [
"VARIANTS = [",
"    (\"omega 15, 60 points\", 15, {}),",
"    (\"omega 30, 60 points\", 30, {}),",
"    (\"omega 60, 60 points\", 60, {}),",
"    (\"omega 60, 240 points\", 60, dict(n_collocation=240)),",
"    (\"omega 60, 240 points, sin x4\", 60, dict(n_collocation=240, activation=\"sin\", input_scale=4.0)),",
"    (\"omega 60, 60 points, sin x4\", 60, dict(activation=\"sin\", input_scale=4.0)),",
"]",
""
],
"note": "<p>The last three settings separate two possible fixes: more places where the equation is checked, and a network that starts out with faster wiggles.</p>"
},
{
"line": 22,
"code": [
"if __name__ == \"__main__\":",
"    jobs = [",
"        (name, seed, options, 0.01, SPRINGS[omega], SPRINGS[omega])",
"        for name, omega, options in VARIANTS",
"        for seed in FEW_SEEDS",
"    ]",
"    rows = run(job, jobs)",
"    truth = {str(w): curve(osc.position) for w, osc in SPRINGS.items()}",
"    save(\"frequency\", {\"truth\": truth, \"rows\": rows})"
],
"note": "<p>Believed and true physics are the same here. Nothing is wrong with the equation, only with how often it is checked.</p>"
}
]
},
{
"path": "experiments/inverse.py",
"title": "Recovering c and k",
"group": "Experiments",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"Recover c and k from the ten dots.",
"",
"    python -m experiments.inverse",
"\"\"\"",
"",
"from pinn.data import measure",
"from pinn.inverse import discover",
"from pinn.models import future_mse",
"from pinn.physics import Oscillator",
"",
"from .common import FEW_SEEDS, curve, run, save",
"",
"OSC = Oscillator()",
"",
""
],
"note": "<p>Three noise levels, three seeds each.</p>"
},
{
"line": 16,
"code": [
"def job(noise, seed):",
"    result = discover(measure(OSC, seed=seed, noise=noise), seed=seed)",
"    row = {",
"        \"noise\": noise,",
"        \"seed\": seed,",
"        \"c\": result.c,",
"        \"k\": result.k,",
"        \"mse\": future_mse(result.predict, OSC),",
"    }",
"    if seed == 0:",
"        row[\"curve\"] = curve(result.predict)",
"        row[\"trace\"] = [[n, round(c, 4), round(k, 3)] for n, c, k in result.trace]",
"    return row",
"",
""
],
"note": "<p>The error is still scored against the true curve, so it can be compared with the forward problem.</p>"
},
{
"line": 31,
"code": [
"if __name__ == \"__main__\":",
"    save(\"inverse\", run(job, [(noise, s) for noise in (0.0, 0.01, 0.03) for s in FEW_SEEDS]))"
],
"note": "<p>The spread over seeds is a rough error bar for each estimate.</p>"
}
]
},
{
"path": "tests/test_classical.py",
"title": "Is the noise floor right?",
"group": "Tests",
"blocks": [
{
"line": 1,
"code": [
"import numpy as np",
"",
"from pinn import classical",
"from pinn.data import future_grid, measure",
"from pinn.physics import Oscillator",
"",
""
],
"note": "<p>The yardstick needs checking too.</p>"
},
{
"line": 8,
"code": [
"def test_perfect_dots_give_a_perfect_future():",
"    osc = Oscillator()",
"    predict = classical.fit(osc, measure(osc, noise=0.0))",
"    np.testing.assert_allclose(predict(future_grid()), osc.position(future_grid()), atol=1e-12)",
"",
""
],
"note": "<p>With no noise, two unknowns and ten equations, the answer must be exact.</p>"
},
{
"line": 14,
"code": [
"def test_predicted_error_matches_simulated_error():",
"    osc, t = Oscillator(), future_grid()",
"    errs = []",
"    for seed in range(2000):",
"        predict = classical.fit(osc, measure(osc, seed=seed))",
"        errs.append(np.mean((predict(t) - osc.position(t)) ** 2))",
"    expected = classical.expected_future_mse(osc, measure(osc), t)",
"    assert abs(np.mean(errs) / expected - 1) < 0.1"
],
"note": "<p>Two thousand noisy data sets, one fit each. Their average error should match the covariance formula. Within 10%, it does.</p>"
}
]
},
{
"path": "tests/test_physics.py",
"title": "Is the reference curve right?",
"group": "Tests",
"blocks": [
{
"line": 1,
"code": [
"import numpy as np",
"import pytest",
"",
"from pinn.physics import Oscillator",
"",
"T = np.linspace(0, 1, 400)",
"",
""
],
"note": "<p>Everything rests on this file.</p>"
},
{
"line": 9,
"code": [
"def central(f, t, h=1e-5):",
"    return (f(t + h) - f(t - h)) / (2 * h)",
"",
""
],
"note": "<p>A central difference. Its error shrinks with h², and it shares no code with the formulas being checked.</p>"
},
{
"line": 13,
"code": [
"def test_starts_where_it_should():",
"    osc = Oscillator(x0=0.7, v0=-2.0)",
"    assert osc.position(0.0) == pytest.approx(0.7)",
"    assert osc.velocity(0.0) == pytest.approx(-2.0)",
"",
""
],
"note": "<p>Uses a non-zero starting velocity, because v0 = 0 hides mistakes in <code>b</code>.</p>"
},
{
"line": 19,
"code": [
"def test_velocity_is_the_slope_of_position():",
"    osc = Oscillator()",
"    np.testing.assert_allclose(osc.velocity(T), central(osc.position, T), atol=1e-6)",
"",
""
],
"note": "<p>Velocity must be the slope of position.</p>"
},
{
"line": 24,
"code": [
"def test_closed_form_satisfies_the_equation():",
"    # acceleration() is defined from the ODE, so compare it with a numerical",
"    # derivative of velocity(). Agreement means the closed form is a solution.",
"    osc = Oscillator()",
"    np.testing.assert_allclose(osc.acceleration(T), central(osc.velocity, T), atol=1e-4)",
"",
""
],
"note": "<p><code>acceleration</code> comes from the ODE and the right-hand side comes from the formula. They agree only if the formula solves the ODE.</p>"
},
{
"line": 31,
"code": [
"def test_energy_never_rises():",
"    energy = Oscillator().energy(T)",
"    assert np.all(np.diff(energy) <= 1e-12)",
"",
"",
"def test_no_friction_keeps_energy_constant():",
"    energy = Oscillator(c=0.0).energy(T)",
"    np.testing.assert_allclose(energy, energy[0], rtol=1e-12)",
"",
""
],
"note": "<p>Physics the formula was never told about. If it holds, the formula is very likely right.</p>"
},
{
"line": 41,
"code": [
"def test_overdamped_is_refused():",
"    with pytest.raises(ValueError):",
"        Oscillator(c=40.0)"
],
"note": "<p>An error now is better than <code>nan</code> later.</p>"
}
]
},
{
"path": "tests/test_data.py",
"title": "Are the dots honest?",
"group": "Tests",
"blocks": [
{
"line": 1,
"code": [
"import numpy as np",
"",
"from pinn.data import T_DATA, T_END, future_grid, measure",
"from pinn.physics import Oscillator",
"",
""
],
"note": "<p>Reproducibility and separation.</p>"
},
{
"line": 7,
"code": [
"def test_same_seed_same_dots():",
"    a, b = measure(Oscillator(), seed=3), measure(Oscillator(), seed=3)",
"    np.testing.assert_array_equal(a.x, b.x)",
"",
"",
"def test_different_seed_different_dots():",
"    a, b = measure(Oscillator(), seed=3), measure(Oscillator(), seed=4)",
"    assert not np.array_equal(a.x, b.x)",
"",
""
],
"note": "<p>Same seed, same data. A different seed must give different data, or the first test could pass on a constant.</p>"
},
{
"line": 17,
"code": [
"def test_every_dot_is_inside_the_window():",
"    m = measure(Oscillator())",
"    assert len(m) == 10",
"    assert m.t.min() == 0.0 and m.t.max() == T_DATA",
"",
""
],
"note": "<p>Every dot is on the left of the wall.</p>"
},
{
"line": 23,
"code": [
"def test_scoring_times_never_touch_the_dots():",
"    grid = future_grid()",
"    assert grid.min() > T_DATA and grid.max() == T_END",
"",
""
],
"note": "<p>Every scoring time is on the right of it.</p>"
},
{
"line": 28,
"code": [
"def test_noise_has_the_advertised_size():",
"    osc = Oscillator()",
"    m = measure(osc, n=20_000, noise=0.05)",
"    assert abs(np.std(m.x - osc.position(m.t)) - 0.05) < 0.002"
],
"note": "<p>Twenty thousand samples, so the estimate of the standard deviation is tight.</p>"
}
]
},
{
"path": "tests/test_derivatives.py",
"title": "Are x' and x'' right?",
"group": "Tests",
"blocks": [
{
"line": 1,
"code": [
"import torch",
"",
"from pinn.derivatives import derivatives",
"from pinn.network import MLP, column",
"",
"T = column(torch.linspace(0.05, 0.95, 40))",
"",
""
],
"note": "<p>Four angles on one short function.</p>"
},
{
"line": 9,
"code": [
"def test_known_function():",
"    x, v, a = derivatives(lambda t: t**3, T)",
"    assert torch.allclose(v, 3 * T**2)",
"    assert torch.allclose(a, 6 * T)",
"",
""
],
"note": "<p>A function whose derivatives anyone can check by eye.</p>"
},
{
"line": 15,
"code": [
"def test_network_against_finite_differences():",
"    net, h = MLP(seed=1), 1e-4",
"    x, v, a = derivatives(net, T)",
"    assert torch.allclose(v, (net(T + h) - net(T - h)) / (2 * h), atol=1e-6)",
"    assert torch.allclose(a, (net(T + h) - 2 * net(T) + net(T - h)) / h**2, atol=1e-5)",
"",
""
],
"note": "<p>A real network, compared with a method that shares no code. The tolerances are loose because finite differences are.</p>"
},
{
"line": 22,
"code": [
"def test_second_derivative_still_reaches_the_weights():",
"    net = MLP(seed=1)",
"    derivatives(net, T)[2].pow(2).mean().backward()",
"    assert net.layers[0].weight.grad.abs().sum() > 0",
"",
""
],
"note": "<p>This is what <code>create_graph=True</code> is for. Remove it and this test fails.</p>"
},
{
"line": 28,
"code": [
"def test_relu_has_no_second_derivative():",
"    net = MLP(activation=\"relu\", seed=1)",
"    assert torch.all(derivatives(net, T)[2] == 0)"
],
"note": "<p>A trap, recorded as a fact.</p>"
}
]
},
{
"path": "tests/test_losses.py",
"title": "Do the losses measure what they claim?",
"group": "Tests",
"blocks": [
{
"line": 1,
"code": [
"import torch",
"",
"from pinn.losses import data_loss, physics_loss, residual",
"from pinn.network import MLP, column",
"from pinn.physics import Oscillator",
"",
"OSC = Oscillator()",
"T = column(torch.linspace(0, 1, 100))",
"",
"",
"def true_motion(t):",
"    d, w = OSC.delta, OSC.omega_d",
"    return torch.exp(-d * t) * (torch.cos(w * t) + d / w * torch.sin(w * t))",
"",
""
],
"note": "<p>The exact solution, written in torch so autograd can differentiate it.</p>"
},
{
"line": 16,
"code": [
"def test_true_motion_has_no_residual():",
"    assert physics_loss(true_motion, T, OSC.m, OSC.c, OSC.k) < 1e-25",
"",
""
],
"note": "<p>The residual of the truth should be zero up to rounding.</p>"
},
{
"line": 20,
"code": [
"def test_wrong_stiffness_shows_up_in_the_residual():",
"    assert physics_loss(true_motion, T, OSC.m, OSC.c, 1.1 * OSC.k) > 1e-4",
"",
""
],
"note": "<p>A 10% error in k must be visible. A loss that can't see wrong physics is useless.</p>"
},
{
"line": 24,
"code": [
"def test_flat_zero_also_satisfies_the_physics():",
"    # The equation alone cannot pick a solution. The dots have to.",
"    assert physics_loss(lambda t: 0 * t, T, OSC.m, OSC.c, OSC.k) == 0",
"",
""
],
"note": "<p>The equation alone can't choose a solution.</p>"
},
{
"line": 29,
"code": [
"def test_residual_is_scaled_to_order_one():",
"    # A constant curve x = 1 breaks the equation by exactly k x / k = 1.",
"    r = residual(lambda t: torch.ones_like(t) + 0 * t, T, OSC.m, OSC.c, OSC.k)",
"    assert torch.allclose(r, torch.ones_like(r))",
"",
""
],
"note": "<p>For x = 1 the derivatives vanish and k/k = 1 remains.</p>"
},
{
"line": 35,
"code": [
"def test_data_loss_is_a_mean_square():",
"    net = MLP(seed=0)",
"    x = net(T).detach() + 0.5",
"    assert torch.allclose(data_loss(net, T, x), torch.tensor(0.25, dtype=torch.float64))"
],
"note": "<p>Shift the target by 0.5 and the mean square must be 0.25.</p>"
}
]
},
{
"path": "tests/test_training.py",
"title": "Does training do its job?",
"group": "Tests",
"blocks": [
{
"line": 1,
"code": [
"import numpy as np",
"",
"from pinn import classical",
"from pinn.data import measure",
"from pinn.models import future_mse, train_pinn, train_plain",
"from pinn.physics import Oscillator",
"",
"OSC = Oscillator()",
"",
""
],
"note": "<p>End-to-end checks, kept short enough to run on every push.</p>"
},
{
"line": 11,
"code": [
"def test_training_lowers_the_loss():",
"    result = train_plain(measure(OSC), adam_steps=300)",
"    assert result.history[-1][\"total\"] < 0.3 * result.history[0][\"total\"]",
"",
""
],
"note": "<p>The weakest claim a training loop has to meet.</p>"
},
{
"line": 16,
"code": [
"def test_same_seed_same_result():",
"    options = dict(adam_steps=50, lbfgs_steps=20)",
"    a = train_pinn(measure(OSC), OSC, **options).predict([0.5])",
"    b = train_pinn(measure(OSC), OSC, **options).predict([0.5])",
"    np.testing.assert_array_equal(a, b)",
"",
""
],
"note": "<p>Bit for bit, on the same machine.</p>"
},
{
"line": 23,
"code": [
"def test_scoring_the_truth_gives_zero():",
"    assert future_mse(OSC.position, OSC) == 0",
"",
""
],
"note": "<p>A score that isn't zero for the right answer would make every comparison meaningless.</p>"
},
{
"line": 27,
"code": [
"def test_physics_beats_no_physics():",
"    # Shortened training, so the bar is loose. The full numbers live in results/.",
"    data = measure(OSC)",
"    plain = future_mse(train_plain(data, adam_steps=1000).predict, OSC)",
"    pinn = future_mse(train_pinn(data, OSC, adam_steps=1000, lbfgs_steps=1000).predict, OSC)",
"    assert pinn < plain / 50"
],
"note": "<p>A shortened version of the headline. The factor of 50 is far below the measured one, so ordinary variation won't break it.</p>"
}
]
},
{
"path": "tests/test_scratch.py",
"title": "Hand-written gradients against autograd",
"group": "Tests",
"blocks": [
{
"line": 1,
"code": [
"\"\"\"The hand-written gradients against autograd.\"\"\"",
"",
"import numpy as np",
"import torch",
"",
"from pinn.data import measure",
"from pinn.derivatives import derivatives",
"from pinn.losses import data_loss, physics_loss",
"from pinn.network import MLP, column",
"from pinn.physics import Oscillator",
"from pinn.scratch import Adam, ScratchMLP, tanh_and_derivatives",
"",
"OSC = Oscillator()",
"T_COL = np.linspace(0, 1, 25)",
"",
""
],
"note": "<p>The test that makes chapter 10 believable.</p>"
},
{
"line": 17,
"code": [
"def twins():",
"    torch_net, scratch = MLP(width=12, seed=2), ScratchMLP(width=12)",
"    linears = [layer for layer in torch_net.layers if hasattr(layer, \"weight\")]",
"    scratch.W = [layer.weight.detach().numpy().copy() for layer in linears]",
"    scratch.b = [layer.bias.detach().numpy().copy() for layer in linears]",
"    return torch_net, linears, scratch",
"",
""
],
"note": "<p>Copy PyTorch's weights into the NumPy network, so both compute the same function.</p>"
},
{
"line": 25,
"code": [
"def test_tanh_derivatives():",
"    z, h = np.linspace(-2, 2, 50), 1e-5",
"    for order in range(3):",
"        numeric = (tanh_and_derivatives(z + h)[order] - tanh_and_derivatives(z - h)[order]) / (2 * h)",
"        np.testing.assert_allclose(tanh_and_derivatives(z)[order + 1], numeric, atol=1e-8)",
"",
""
],
"note": "<p>Each derivative is the slope of the one before it.</p>"
},
{
"line": 32,
"code": [
"def test_forward_matches_autograd():",
"    torch_net, _, scratch = twins()",
"    ours = scratch.forward(T_COL)[:3]",
"    for mine, theirs in zip(ours, derivatives(torch_net, column(T_COL))):",
"        np.testing.assert_allclose(mine, theirs.detach().numpy(), atol=1e-11)",
"",
""
],
"note": "<p>x, x' and x'' from forward propagation against two reverse passes of autograd.</p>"
},
{
"line": 39,
"code": [
"def test_every_weight_gradient_matches_autograd():",
"    torch_net, linears, scratch = twins()",
"    data = measure(OSC)",
"    loss = data_loss(torch_net, column(data.t), column(data.x))",
"    loss = loss + 0.3 * physics_loss(torch_net, column(T_COL), OSC.m, OSC.c, OSC.k)",
"    loss.backward()",
"    theirs = [layer.weight.grad.numpy() for layer in linears]",
"    theirs += [layer.bias.grad.numpy() for layer in linears]",
"    _, ours = scratch.loss_and_grads(data.t, data.x, T_COL, OSC.m, OSC.c, OSC.k, weight=0.3)",
"    assert len(ours) == len(theirs) == 8",
"    for mine, other in zip(ours, theirs):",
"        np.testing.assert_allclose(mine, other, rtol=1e-9, atol=1e-12)",
"",
""
],
"note": "<p>Eight arrays, every entry, to a relative tolerance of 1e-9.</p>"
},
{
"line": 53,
"code": [
"def test_adam_matches_torch_adam():",
"    start, grad = np.array([1.0, -2.0]), np.array([0.3, -0.1])",
"    mine = start.copy()",
"    ours = Adam([mine], lr=0.01)",
"    theirs = torch.tensor(start, requires_grad=True)",
"    optimiser = torch.optim.Adam([theirs], lr=0.01)",
"    for _ in range(5):",
"        ours.step([grad])",
"        theirs.grad = torch.tensor(grad)",
"        optimiser.step()",
"    np.testing.assert_allclose(mine, theirs.detach().numpy(), rtol=1e-9)"
],
"note": "<p>Five steps with a fixed gradient. The bias correction is the part that is easy to get wrong.</p>"
}
]
},
{
"path": "tests/engine.test.mjs",
"title": "The browser engine against NumPy",
"group": "Tests",
"blocks": [
{
"line": 1,
"code": [
"// node --test tests/engine.test.mjs",
"import { readFileSync } from \"node:fs\";",
"import { createRequire } from \"node:module\";",
"import test from \"node:test\";",
"import assert from \"node:assert\";",
"",
"const require = createRequire(import.meta.url);",
"const Engine = require(\"../docs/assets/engine.js\");",
"const fx = JSON.parse(readFileSync(new URL(\"./engine_fixture.json\", import.meta.url)));",
"const osc = Engine.oscillator(1, 3, 225);",
"const near = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) <= eps * (1 + Math.abs(b)), `${a} vs ${b}`);",
""
],
"note": "<p>No packages. Node's own test runner is enough.</p>"
},
{
"line": 13,
"code": [
"function net() {",
"  const n = new Engine.Net({ width: 6, depth: 3 });",
"  fx.W.forEach((w, i) => n.W[i].set(w));",
"  fx.b.forEach((b, i) => n.b[i].set(b));",
"  return n;",
"}",
""
],
"note": "<p>Load the weights that <code>tools/make_engine_fixture.py</code> wrote.</p>"
},
{
"line": 20,
"code": [
"test(\"closed form matches Python\", () => {",
"  for (const [t, x, v] of fx.truth) { near(osc.x(t), x); near(osc.v(t), v); }",
"});",
"",
"test(\"loss matches NumPy\", () => {",
"  const parts = net().lossAndGrads(fx.data, fx.t_col, osc, 0.3);",
"  near(parts.data, fx.parts.data); near(parts.physics, fx.parts.physics);",
"});",
""
],
"note": "<p>Two implementations of the same formula, in two languages.</p>"
},
{
"line": 29,
"code": [
"test(\"every gradient matches NumPy\", () => {",
"  const n = net();",
"  n.lossAndGrads(fx.data, fx.t_col, osc, 0.3);",
"  fx.gW.forEach((g, l) => g.forEach((v, i) => near(n.gW[l][i], v)));",
"  fx.gb.forEach((g, l) => g.forEach((v, i) => near(n.gb[l][i], v)));",
"});",
""
],
"note": "<p>If this passes, the figures on the site train the same model as the Python code.</p>"
},
{
"line": 36,
"code": [
"test(\"training lowers the future error\", () => {",
"  const data = Engine.measure(osc, { seed: 1 });",
"  const run = new Engine.Trainer({ osc, data, steps: 4000, seed: 1 });",
"  run.advance(4000);",
"  const mse = run.futureMse();",
"  console.log(\"future mse\", mse, run.parts);",
"  assert.ok(mse < 5e-3);",
"});"
],
"note": "<p>A smoke test: a full training run must end with a small error.</p>"
}
]
}
];
