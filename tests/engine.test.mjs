// node --test tests/engine.test.mjs
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import assert from "node:assert";

const require = createRequire(import.meta.url);
const Engine = require("../docs/assets/engine.js");
const fx = JSON.parse(readFileSync(new URL("./engine_fixture.json", import.meta.url)));
const osc = Engine.oscillator(1, 3, 225);
const near = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) <= eps * (1 + Math.abs(b)), `${a} vs ${b}`);

function net() {
  const n = new Engine.Net({ width: 6, depth: 3 });
  fx.W.forEach((w, i) => n.W[i].set(w));
  fx.b.forEach((b, i) => n.b[i].set(b));
  return n;
}

test("closed form matches Python", () => {
  for (const [t, x, v] of fx.truth) { near(osc.x(t), x); near(osc.v(t), v); }
});

test("loss matches NumPy", () => {
  const parts = net().lossAndGrads(fx.data, fx.t_col, osc, 0.3);
  near(parts.data, fx.parts.data); near(parts.physics, fx.parts.physics);
});

test("every gradient matches NumPy", () => {
  const n = net();
  n.lossAndGrads(fx.data, fx.t_col, osc, 0.3);
  fx.gW.forEach((g, l) => g.forEach((v, i) => near(n.gW[l][i], v)));
  fx.gb.forEach((g, l) => g.forEach((v, i) => near(n.gb[l][i], v)));
});

test("training lowers the future error", () => {
  const data = Engine.measure(osc, { seed: 1 });
  const run = new Engine.Trainer({ osc, data, steps: 4000, seed: 1 });
  run.advance(4000);
  const mse = run.futureMse();
  console.log("future mse", mse, run.parts);
  assert.ok(mse < 5e-3);
});
